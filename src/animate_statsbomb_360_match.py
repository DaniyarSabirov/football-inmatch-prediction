"""Visualize event-based StatsBomb 360 freeze frames for one match segment."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, colors as matplotlib_colors
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, Rectangle
from PIL import Image
from scipy.optimize import linear_sum_assignment

try:
    # Flat sibling import: how this module sees it when run directly as a
    # script (e.g. `python src/animate_statsbomb_360_match.py`), which puts
    # this file's own directory (src/) at sys.path[0].
    from orientation_correction import (
        ORIENTATION_NORMAL,
        mirror_point,
        ORIENTATION_ROTATED_180,
        ORIENTATION_UNCERTAIN,
        OrientationResult,
        apply_correction,
        classify_orientations_for_match,
    )
    from team_normalization import (
        TEAM_NORMALIZATION_MIRRORED,
        TEAM_NORMALIZATION_REFERENCE,
        apply_team_normalization_to_event,
        apply_team_normalization_to_frame,
        resolve_reference_team_id,
        resolve_physical_reference_team_id,
        apply_period_end_swap,
        rotate_event_180,
    )
    from physical_orientation_calibration import (
        load_calibration_registry,
        load_frame_reviews,
        load_period_reviews,
        resolve_usable_physical_reference_team_name,
    )
    from broadcast_orientation import (
        PROVENANCE_PREFERENCE,
        PROVENANCE_VERIFIED,
        BroadcastOrientationInference,
        DisplayOrientationPlan,
        build_display_plan,
        CAMERA_SIDE_UNKNOWN,
        estimate_camera_side,
        expected_own_frame_camera_side,
        infer_broadcast_orientation,
        kickoff_own_half_orientation,
        resolve_uncertain_for_display,
    )
    # Orientation resolution shared with the dataset builder (moved there
    # unchanged on 2026-10-26); re-exported here for existing scripts.
    from orientation_resolution import (
        CONTINUITY_MAX_SECONDS,
        CONTINUITY_RATIO,
        display_orientation_results,
        frame_review_corrections,
        infer_broadcast_orientation_for_match,
        resolve_display_plan,
    )
    from team_identity import (
        IDENTITY_CORRECTED_GOALKEEPER,
        IDENTITY_CORRECTED_VIEW,
        FrameIdentity,
        resolve_match_team_identity,
        snapshot_key,
        TEAM_IDENTITY_AMBIGUOUS,
        SharedGroupInfo,
        detect_shared_cross_team_snapshots,
        resolve_absolute_team_for_point,
    )
except ImportError:
    # Package-style import: how this module sees it when run as
    # `python -m src.animate_statsbomb_360_match` from the project root,
    # which puts the project root (not src/) at sys.path[0].
    from src.orientation_correction import (
        ORIENTATION_NORMAL,
        mirror_point,
        ORIENTATION_ROTATED_180,
        ORIENTATION_UNCERTAIN,
        OrientationResult,
        apply_correction,
        classify_orientations_for_match,
    )
    from src.team_normalization import (
        TEAM_NORMALIZATION_MIRRORED,
        TEAM_NORMALIZATION_REFERENCE,
        apply_team_normalization_to_event,
        apply_team_normalization_to_frame,
        resolve_reference_team_id,
        resolve_physical_reference_team_id,
        apply_period_end_swap,
        rotate_event_180,
    )
    from src.physical_orientation_calibration import (
        load_calibration_registry,
        load_frame_reviews,
        load_period_reviews,
        resolve_usable_physical_reference_team_name,
    )
    from src.broadcast_orientation import (
        PROVENANCE_PREFERENCE,
        PROVENANCE_VERIFIED,
        BroadcastOrientationInference,
        DisplayOrientationPlan,
        build_display_plan,
        CAMERA_SIDE_UNKNOWN,
        estimate_camera_side,
        expected_own_frame_camera_side,
        infer_broadcast_orientation,
        kickoff_own_half_orientation,
        resolve_uncertain_for_display,
    )
    # Orientation resolution shared with the dataset builder (moved there
    # unchanged on 2026-10-26); re-exported here for existing scripts.
    from src.orientation_resolution import (
        CONTINUITY_MAX_SECONDS,
        CONTINUITY_RATIO,
        display_orientation_results,
        frame_review_corrections,
        infer_broadcast_orientation_for_match,
        resolve_display_plan,
    )
    from src.team_identity import (
        IDENTITY_CORRECTED_GOALKEEPER,
        IDENTITY_CORRECTED_VIEW,
        FrameIdentity,
        resolve_match_team_identity,
        snapshot_key,
        TEAM_IDENTITY_AMBIGUOUS,
        SharedGroupInfo,
        detect_shared_cross_team_snapshots,
        resolve_absolute_team_for_point,
    )

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATSBOMB_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
DEFAULT_MATCH_ID = 3902240
DEFAULT_PERIOD = 1
DEFAULT_START_MINUTE = 10
DEFAULT_END_MINUTE = 20
DEFAULT_FRAME_DURATION_MS = 250
FIXED_X_LIMITS = (-3.0, 123.0)
FIXED_Y_LIMITS = (83.0, -3.0)
# Inside the 3-unit margin beyond the far touchline, so goal labels never
# collide with the secondary header line drawn just above the axes.
GOAL_LABEL_Y = -1.5
VIEW_ALIASES = {
    "analysis": "analysis",
    "camera-overlay": "camera-overlay",
    "full-pitch": "analysis",
    "visible-area": "camera-overlay",
}
TEAM_DISPLAY_LABELS = {
    "Netherlands Women's": "NL",
    "Spain Women's": "ES",
    "Manchester City": "MCI",
    "Inter Milan": "INT",
    "Inter": "INT",
}


class VisualizationError(Exception):
    """Raised for an input or output problem that should be shown to the user."""


@dataclass(frozen=True)
class JoinedFrame:
    """One event joined to its corresponding StatsBomb 360 record."""

    event: dict[str, Any]
    frame_360: dict[str, Any]


@dataclass(frozen=True)
class MatchFrames:
    """Loaded match records and join counts."""

    event_path: Path
    three_sixty_path: Path
    event_count: int
    three_sixty_count: int
    matched_count: int
    usable_count: int
    usable_frames: list[JoinedFrame]


@dataclass(frozen=True)
class MatchMetadata:
    """Optional StatsBomb match metadata used for discovery and summaries."""

    match_id: int
    date: str
    competition: str
    season: str
    home_team: str
    away_team: str
    home_score: int | None
    away_score: int | None

    @property
    def score(self) -> str:
        """Return a readable score or an unknown marker."""
        if self.home_score is None or self.away_score is None:
            return "—"
        return f"{self.home_score}-{self.away_score}"


@dataclass(frozen=True)
class StructuralReport:
    """Results of validating selected raw coordinates without changing them."""

    player_coordinate_count: int
    player_outside_count: int
    player_outside_frame_count: int
    ball_coordinate_count: int
    ball_outside_count: int
    visible_area_count: int
    visible_area_odd_count: int
    polygon_point_count: int
    polygon_outside_count: int


@dataclass(frozen=True)
class SegmentSummary:
    """Descriptive statistics for an event-based segment."""

    first_timestamp: str
    last_timestamp: str
    average_gap: float
    minimum_gap: float
    median_gap: float
    maximum_gap: float
    average_visible_players: float
    minimum_visible_players: int
    maximum_visible_players: int
    valid_ball_frames: int
    valid_polygon_frames: int


@dataclass(frozen=True)
class VisualTheme:
    """Centralized colors for one presentation theme."""

    name: str
    figure_background: str
    pitch_background: str
    stripe_colors: tuple[str, str]
    pitch_lines: str
    primary_text: str
    secondary_text: str
    marker_edge: str
    legend_face: str
    legend_edge: str
    actor_accent: str
    correction_ring: str
    ball_fill: str
    ball_edge: str
    camera_fill: str
    camera_edge: str
    camera_alpha: float
    default_team_one: str
    default_team_two: str
    ambiguous_fill: str


THEMES = {
    "dark": VisualTheme(
        name="dark",
        figure_background="#08110D",
        pitch_background="#0D1A15",
        stripe_colors=("#0D1A15", "#101F19"),
        pitch_lines="#53645D",
        primary_text="#E8EFEA",
        secondary_text="#A8B5AE",
        marker_edge="#06100C",
        legend_face="#0A1511",
        legend_edge="#53645D",
        actor_accent="#F4D35E",
        correction_ring="#7FE7DC",
        ball_fill="#FFFFFF",
        ball_edge="#08110D",
        camera_fill="#F4D35E",
        camera_edge="#E6C44A",
        camera_alpha=0.13,
        default_team_one="#27A7E8",
        default_team_two="#FF5A5F",
        ambiguous_fill="#8A8F94",
    ),
    "light": VisualTheme(
        name="light",
        figure_background="#F4F6F4",
        pitch_background="#3F8F52",
        stripe_colors=("#3F8F52", "#438F55"),
        pitch_lines="#D7DED8",
        primary_text="#17211C",
        secondary_text="#526159",
        marker_edge="#17212B",
        legend_face="#F7FAF8",
        legend_edge="#8A9991",
        actor_accent="#E5B82E",
        correction_ring="#0E8C82",
        ball_fill="#FFFFFF",
        ball_edge="#111111",
        camera_fill="#FFE9A6",
        camera_edge="#D2B85C",
        camera_alpha=0.20,
        default_team_one="#2474B5",
        default_team_two="#D64545",
        ambiguous_fill="#9AA0A6",
    ),
}


def get_visual_theme(name: str) -> VisualTheme:
    """Return a named visual theme."""
    return THEMES[name]


@dataclass(frozen=True)
class TeamStyle:
    """Stable match-level team names and colors."""

    team_one: str
    team_two: str
    team_one_color: str
    team_two_color: str
    team_one_label: str
    team_two_label: str

    def color_for(self, team_name: str) -> str:
        """Return the color permanently assigned to a match team."""
        return self.team_one_color if team_name == self.team_one else self.team_two_color

    def opponent_of(self, team_name: str) -> str:
        """Return the other match team."""
        return self.team_two if team_name == self.team_one else self.team_one


@dataclass(frozen=True)
class DirectionInference:
    """Conservative period-specific attacking-direction result."""

    directions: dict[str, int]
    explanation: str
    # One-line, on-figure statement of how the display orientation was
    # established (verified / inferred / unknown); empty draws nothing.
    provenance: str = ""


@dataclass(frozen=True)
class FrameTiming:
    """Real timestamp gaps and chosen GIF hold duration for one actual event."""

    match_second: float
    gap_from_previous: float | None
    gap_to_next: float | None
    display_duration_ms: int


@dataclass(frozen=True)
class TemporalDensity:
    """Temporal-density statistics for event-based freeze frames."""

    requested_seconds: float
    frame_count: int
    frames_per_second: float
    average_gap: float | None
    minimum_gap: float | None
    percentile_25: float | None
    median_gap: float | None
    percentile_75: float | None
    maximum_gap: float | None
    bins: tuple[tuple[str, int, float], ...]


@dataclass(frozen=True)
class EventSelection:
    """A selection containing only real joined StatsBomb 360 events."""

    frames: list[JoinedFrame]
    mode: str
    period: int | None
    interval_start: float | None
    interval_end: float | None
    interval_label: str


def raw_input_paths(match_id: int) -> tuple[Path, Path]:
    """Return exact event and 360 paths for a StatsBomb match ID."""
    return (
        STATSBOMB_DATA_DIR / "events" / f"{match_id}.json",
        STATSBOMB_DATA_DIR / "three-sixty" / f"{match_id}.json",
    )


def available_360_match_ids() -> list[int]:
    """Return match IDs that have a StatsBomb three-sixty JSON file."""
    match_ids = []
    for path in (STATSBOMB_DATA_DIR / "three-sixty").glob("*.json"):
        if path.stem.isdigit():
            match_ids.append(int(path.stem))
    return sorted(match_ids)


def metadata_from_record(record: dict[str, Any]) -> MatchMetadata | None:
    """Normalize one StatsBomb match metadata record."""
    match_id = record.get("match_id")
    if isinstance(match_id, bool) or not isinstance(match_id, int):
        return None

    def object_name(field: str, name_field: str) -> str:
        value = record.get(field)
        if isinstance(value, dict) and isinstance(value.get(name_field), str):
            return value[name_field]
        return "Unknown"

    home_score = record.get("home_score")
    away_score = record.get("away_score")
    return MatchMetadata(
        match_id=match_id,
        date=str(record.get("match_date", "Unknown")),
        competition=object_name("competition", "competition_name"),
        season=object_name("season", "season_name"),
        home_team=object_name("home_team", "home_team_name"),
        away_team=object_name("away_team", "away_team_name"),
        home_score=home_score if isinstance(home_score, int) else None,
        away_score=away_score if isinstance(away_score, int) else None,
    )


def load_match_metadata_index() -> dict[int, MatchMetadata]:
    """Load optional metadata for all locally available StatsBomb matches."""
    index: dict[int, MatchMetadata] = {}
    for path in sorted((STATSBOMB_DATA_DIR / "matches").glob("**/*.json")):
        try:
            records = load_json_records(path, "match metadata")
        except VisualizationError as error:
            print(f"Warning: {error}", file=sys.stderr)
            continue
        for record in records:
            metadata = metadata_from_record(record)
            if metadata is not None:
                index[metadata.match_id] = metadata
    return index


def text_matches(value: str, query: str | None) -> bool:
    """Apply a case-insensitive exact-or-substring metadata filter."""
    return query is None or query.casefold() in value.casefold()


def filter_available_matches(
    metadata_index: dict[int, MatchMetadata],
    team_filter: str | None = None,
    competition_filter: str | None = None,
    season_filter: str | None = None,
) -> list[tuple[int, MatchMetadata | None]]:
    """Filter matches that have 360 data using optional metadata text."""
    results = []
    for match_id in available_360_match_ids():
        metadata = metadata_index.get(match_id)
        if team_filter is not None and (
            metadata is None
            or not (
                text_matches(metadata.home_team, team_filter)
                or text_matches(metadata.away_team, team_filter)
            )
        ):
            continue
        if competition_filter is not None and (
            metadata is None
            or not text_matches(metadata.competition, competition_filter)
        ):
            continue
        if season_filter is not None and (
            metadata is None or not text_matches(metadata.season, season_filter)
        ):
            continue
        results.append((match_id, metadata))
    return results


def format_match_candidate(match_id: int, metadata: MatchMetadata | None) -> str:
    """Format one match candidate for listing and resolution errors."""
    if metadata is None:
        return f"{match_id}: metadata unavailable"
    return (
        f"{match_id}: {metadata.date} | {metadata.home_team} vs "
        f"{metadata.away_team} | {metadata.competition} | {metadata.season} | "
        f"{metadata.score}"
    )


def resolve_match_id(
    explicit_match_id: int | None,
    metadata_index: dict[int, MatchMetadata],
    home_team: str | None,
    away_team: str | None,
    match_date: str | None,
) -> int:
    """Resolve a unique available-360 match, preferring an explicit ID."""
    if explicit_match_id is not None:
        return explicit_match_id
    if home_team is None and away_team is None and match_date is None:
        return DEFAULT_MATCH_ID

    candidates = []
    for match_id in available_360_match_ids():
        metadata = metadata_index.get(match_id)
        if metadata is None:
            continue
        if not text_matches(metadata.home_team, home_team):
            continue
        if not text_matches(metadata.away_team, away_team):
            continue
        if not text_matches(metadata.date, match_date):
            continue
        candidates.append((match_id, metadata))
    if not candidates:
        raise VisualizationError("No available 360 match matches the requested teams/date.")
    if len(candidates) > 1:
        details = "\n".join(format_match_candidate(*candidate) for candidate in candidates)
        raise VisualizationError(
            "Match resolution is ambiguous; specify --match-id. Candidates:\n"
            f"{details}"
        )
    return candidates[0][0]


def load_json_records(path: Path, label: str) -> list[dict[str, Any]]:
    """Load a JSON list whose entries must all be objects."""
    try:
        with path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except JSONDecodeError as error:
        raise VisualizationError(f"Invalid JSON in {label} file {path}: {error}") from error
    except OSError as error:
        raise VisualizationError(f"Could not read {label} file {path}: {error}") from error

    if not isinstance(payload, list):
        raise VisualizationError(f"Invalid {label} file {path}: expected a JSON list.")
    if not all(isinstance(record, dict) for record in payload):
        raise VisualizationError(
            f"Invalid {label} file {path}: every list item must be an object."
        )
    return payload


def load_and_join_match(match_id: int) -> MatchFrames:
    """Load event and 360 files, join UUIDs, and keep non-empty freeze frames."""
    event_path, three_sixty_path = raw_input_paths(match_id)
    if not event_path.is_file():
        raise VisualizationError(f"Missing event file: {event_path}")
    if not three_sixty_path.is_file():
        raise VisualizationError(f"Missing three-sixty file: {three_sixty_path}")

    events = load_json_records(event_path, "event")
    frames_360 = load_json_records(three_sixty_path, "three-sixty")

    events_by_uuid: dict[str, dict[str, Any]] = {}
    for event in events:
        event_uuid = event.get("id")
        if not isinstance(event_uuid, str) or not event_uuid:
            raise VisualizationError(f"Invalid event file {event_path}: event missing id.")
        if event_uuid in events_by_uuid:
            raise VisualizationError(
                f"Invalid event file {event_path}: duplicate event UUID {event_uuid}."
            )
        events_by_uuid[event_uuid] = event

    matched: list[JoinedFrame] = []
    seen_frame_uuids: set[str] = set()
    for frame in frames_360:
        event_uuid = frame.get("event_uuid")
        if not isinstance(event_uuid, str) or not event_uuid:
            raise VisualizationError(
                f"Invalid three-sixty file {three_sixty_path}: record missing event_uuid."
            )
        if event_uuid in seen_frame_uuids:
            raise VisualizationError(
                "Invalid three-sixty file "
                f"{three_sixty_path}: duplicate event_uuid {event_uuid}."
            )
        seen_frame_uuids.add(event_uuid)
        event = events_by_uuid.get(event_uuid)
        if event is not None:
            matched.append(JoinedFrame(event=event, frame_360=frame))

    if not matched:
        raise VisualizationError(
            "No matching UUIDs between event and three-sixty files for "
            f"match {match_id}."
        )

    usable = [
        joined
        for joined in matched
        if isinstance(joined.frame_360.get("freeze_frame"), list)
        and bool(joined.frame_360["freeze_frame"])
    ]
    if not usable:
        raise VisualizationError(
            f"Missing freeze_frame: match {match_id} has no non-empty freeze frames."
        )

    usable.sort(key=chronological_key)
    return MatchFrames(
        event_path=event_path,
        three_sixty_path=three_sixty_path,
        event_count=len(events),
        three_sixty_count=len(frames_360),
        matched_count=len(matched),
        usable_count=len(usable),
        usable_frames=usable,
    )


def compute_orientation_results(match: MatchFrames) -> list[OrientationResult]:
    """Classify every usable 360 snapshot's orientation, in chronological order.

    Uses src/orientation_correction.py -- the single source of truth shared
    with the dataset builder -- so visualization and dataset generation apply
    identical logic.
    """
    frames = match.usable_frames
    events = [joined.event for joined in frames]
    frames_360 = [joined.frame_360 for joined in frames]
    timestamps = [timestamp_seconds(event.get("timestamp")) for event in events]
    periods = [integer_event_field(event, "period") for event in events]
    return classify_orientations_for_match(events, frames_360, timestamps, periods)


def apply_orientation_correction_to_match(
    match: MatchFrames, orientation_results: list[OrientationResult]
) -> MatchFrames:
    """Correct every usable 360 snapshot using its precomputed classification.

    NORMAL and UNCERTAIN frames pass through unchanged; for a ROTATED_180
    frame, the ENTIRE snapshot (every freeze_frame player, including
    actor/goalkeeper points, plus the visible_area polygon) is rotated
    together, never a single point in isolation.
    """
    corrected_frames = [
        JoinedFrame(
            event=joined.event,
            frame_360=apply_correction(joined.frame_360, result.orientation),
        )
        for joined, result in zip(match.usable_frames, orientation_results)
    ]
    return MatchFrames(
        event_path=match.event_path,
        three_sixty_path=match.three_sixty_path,
        event_count=match.event_count,
        three_sixty_count=match.three_sixty_count,
        matched_count=match.matched_count,
        usable_count=match.usable_count,
        usable_frames=corrected_frames,
    )


def summarize_orientation_results(orientation_results: list[OrientationResult]) -> dict[str, int]:
    """Count how many frames fell into each orientation category."""
    counts = {ORIENTATION_NORMAL: 0, ORIENTATION_ROTATED_180: 0, ORIENTATION_UNCERTAIN: 0}
    for result in orientation_results:
        counts[result.orientation] += 1
    return counts


def apply_team_normalization_to_match(match: MatchFrames, reference_team_id: int) -> MatchFrames:
    """Place both teams' events onto one shared pitch coordinate system (Problem B).

    Independent of, and applied AFTER, apply_orientation_correction_to_match
    (Problem A) -- see src/team_normalization.py for the forensic evidence.
    Problem A asks whether one event's own freeze_frame agrees with its own
    event.location; this asks a different question -- whether two different
    teams' events sit on the same coordinate scale. Every event belonging to
    the non-reference team has its event.location (and end_location), every
    freeze_frame player, and visible_area mirrored together; the reference
    team's events are returned unchanged.
    """
    normalized_frames = [
        JoinedFrame(
            event=apply_team_normalization_to_event(joined.event, reference_team_id),
            frame_360=apply_team_normalization_to_frame(
                joined.frame_360, joined.event, reference_team_id
            ),
        )
        for joined in match.usable_frames
    ]
    return MatchFrames(
        event_path=match.event_path,
        three_sixty_path=match.three_sixty_path,
        event_count=match.event_count,
        three_sixty_count=match.three_sixty_count,
        matched_count=match.matched_count,
        usable_count=match.usable_count,
        usable_frames=normalized_frames,
    )


def apply_team_normalization_to_match_with_physical_calibration(
    match: MatchFrames,
    all_events: list[dict[str, Any]],
    physical_reference_team_name: str,
) -> tuple[MatchFrames, dict[int, int]]:
    """Like apply_team_normalization_to_match, but resolves the reference
    team SEPARATELY FOR EACH EVENT'S OWN PERIOD via
    resolve_physical_reference_team_id, correctly handling the real
    halftime (and extra-time-half) end swap -- see
    docs/full_physical_pitch_orientation_audit.md for the forensic
    evidence that a single, period-independent reference (as
    apply_team_normalization_to_match alone provides) renders the second
    half backwards relative to a real broadcast camera.

    Returns the transformed match AND the period -> reference_team_id
    mapping actually used, so callers can report it for transparency
    (which team was treated as reference in which period is otherwise
    invisible from the output alone).

    Only used when --physical-reference-team is supplied. The default
    analytical path (apply_team_normalization_to_match with one fixed
    reference_team_id) is completely unchanged, still period-independent,
    and is what src/build_statsbomb_360_dataset.py continues to use.
    """
    periods_present = sorted(
        {integer_event_field(joined.event, "period") for joined in match.usable_frames}
    )
    reference_by_period = {
        period: resolve_physical_reference_team_id(all_events, physical_reference_team_name, period)
        for period in periods_present
    }
    return _apply_team_normalization_per_period(match, reference_by_period), reference_by_period


def apply_team_normalization_to_match_spectator_facing(
    match: MatchFrames,
    all_events: list[dict[str, Any]],
) -> tuple[MatchFrames, dict[int, int]]:
    """The renderer's DEFAULT, uncalibrated presentation path -- period-aware
    like apply_team_normalization_to_match_with_physical_calibration, but
    using the automatic, UNVERIFIED resolve_reference_team_id() pick as the
    period-1 baseline instead of an externally-verified team name.

    Root-cause fix for the 2026-10-22 finding (Iceland vs France, match
    3835342): the ORIGINAL default rendering path
    (apply_team_normalization_to_match, still used only when this function
    is not) resolved ONE reference team for the WHOLE match and never
    re-examined it per period -- correct for periods sharing period 1's
    arrangement, systematically backwards for periods after a real
    halftime/extra-time-half end swap (2 and 4), EVEN WITHOUT any physical
    calibration, because "teams swap ends at halftime" is a fact about the
    sport, not about any specific camera. See
    docs/full_visualization_orientation_and_clock_audit_2026-10-22.md.

    This does NOT constitute physical verification -- the period-1
    baseline is still the same arbitrary (if deterministic) automatic
    pick as before, with no connection to real stadium/camera geometry
    (see resolve_reference_team_id's own docstring). It only guarantees
    that whichever arbitrary side is chosen for period 1 is applied
    CONSISTENTLY with the real, known structure of a football match
    across all periods -- a presentation-layer correctness property,
    independent of and not requiring physical calibration. Explicit
    --physical-reference-team or a verified registry entry still take
    priority whenever available (see main()).
    """
    periods_present = sorted(
        {integer_event_field(joined.event, "period") for joined in match.usable_frames}
    )
    automatic_baseline = resolve_reference_team_id(all_events)
    reference_by_period = {
        period: apply_period_end_swap(all_events, automatic_baseline, period)
        for period in periods_present
    }
    return _apply_team_normalization_per_period(match, reference_by_period), reference_by_period


def _apply_team_normalization_per_period(
    match: MatchFrames, reference_by_period: dict[int, int]
) -> MatchFrames:
    """Shared core: apply team normalization using a precomputed
    period -> reference_team_id mapping, regardless of how that mapping
    was derived (physical calibration or the automatic spectator-facing
    baseline). Never called with a mapping that omits a present period.
    """
    normalized_frames = [
        JoinedFrame(
            event=apply_team_normalization_to_event(
                joined.event,
                reference_by_period[integer_event_field(joined.event, "period")],
            ),
            frame_360=apply_team_normalization_to_frame(
                joined.frame_360,
                joined.event,
                reference_by_period[integer_event_field(joined.event, "period")],
            ),
        )
        for joined in match.usable_frames
    ]
    return MatchFrames(
        event_path=match.event_path,
        three_sixty_path=match.three_sixty_path,
        event_count=match.event_count,
        three_sixty_count=match.three_sixty_count,
        matched_count=match.matched_count,
        usable_count=match.usable_count,
        usable_frames=normalized_frames,
    )


def match_team_ids(match: MatchFrames) -> tuple[int, int]:
    """The match's two team ids, in chronological order of first appearance."""
    team_ids: list[int] = []
    for joined in match.usable_frames:
        team = joined.event.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        if isinstance(team_id, int) and not isinstance(team_id, bool) and team_id not in team_ids:
            team_ids.append(team_id)
    if len(team_ids) != 2:
        raise VisualizationError(f"Expected exactly two team ids, found {team_ids}.")
    return team_ids[0], team_ids[1]


def apply_display_plan_to_match(match: MatchFrames, plan: DisplayOrientationPlan) -> MatchFrames:
    """Apply the display-only orientation layer (Problem D) to every frame.

    Each event and its whole snapshot (every freeze_frame player and the
    visible_area polygon) are rotated together, or not at all, exactly as
    `plan` says for that event's (period, team) -- so players, ball and
    polygon can never be transformed differently. Replaces team
    normalization for display; the analytical pipeline is unaffected.
    """
    displayed = []
    for joined in match.usable_frames:
        team = joined.event.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        if plan.rotates(integer_event_field(joined.event, "period"), team_id):
            displayed.append(
                JoinedFrame(
                    event=rotate_event_180(joined.event),
                    frame_360=apply_correction(joined.frame_360, ORIENTATION_ROTATED_180),
                )
            )
        else:
            displayed.append(JoinedFrame(event=dict(joined.event), frame_360=dict(joined.frame_360)))
    return MatchFrames(
        event_path=match.event_path,
        three_sixty_path=match.three_sixty_path,
        event_count=match.event_count,
        three_sixty_count=match.three_sixty_count,
        matched_count=match.matched_count,
        usable_count=match.usable_count,
        usable_frames=displayed,
    )


def summarize_display_plan(match: MatchFrames, plan: DisplayOrientationPlan) -> dict[str, int]:
    """Count unrotated (REFERENCE) vs rotated (MIRRORED) usable frames under `plan`."""
    counts = {TEAM_NORMALIZATION_REFERENCE: 0, TEAM_NORMALIZATION_MIRRORED: 0}
    for joined in match.usable_frames:
        team = joined.event.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        rotated = plan.rotates(integer_event_field(joined.event, "period"), team_id)
        counts[TEAM_NORMALIZATION_MIRRORED if rotated else TEAM_NORMALIZATION_REFERENCE] += 1
    return counts


PROVENANCE_DISPLAY_TEXT = {
    "VERIFIED": "verified against broadcast evidence (calibration registry / explicit flag)",
    "PREFERENCE": "explicit presentation preference - NOT broadcast-verified",
    "INFERRED": "inferred from StatsBomb visible_area camera side - not broadcast-verified",
    "PARTIALLY_INFERRED": "inferred from visible_area, some periods by halftime rule - not broadcast-verified",
    "UNKNOWN": "deterministic fallback - physical orientation UNKNOWN",
    "USER_CONFIRMED": "confirmed by user review of renders - not independent video verification",
    "USER_CONFIRMED_AND_DERIVED": "user-reviewed period + halftime-rule derivation - not independent video verification",
}


PERIOD_SOURCE_DISPLAY_TEXT = {
    "user_confirmed": "P{period} confirmed by user review of renders - not independent video verification",
    "laws_rule": "P{period} derived by halftime rule from another period - not broadcast-verified",
    "derived_from_user_reviewed_period_and_halftime_rule": (
        "P{period} derived from a user-reviewed period by the halftime rule - not broadcast-verified"
    ),
}


def orientation_provenance_text(plan: DisplayOrientationPlan | None, period: int | None = None) -> str:
    """The on-figure provenance line for a display plan (empty without one).

    With `period`, a period-specific source (user-confirmed, rule-derived)
    is stated instead of the match-level provenance, so a frame never
    borrows a stronger claim from another period.
    """
    if plan is None:
        return ""
    transform = plan.transforms.get(period) if period is not None else None
    if transform is not None and transform.source in PERIOD_SOURCE_DISPLAY_TEXT:
        detail = PERIOD_SOURCE_DISPLAY_TEXT[transform.source].format(period=period)
    else:
        detail = PROVENANCE_DISPLAY_TEXT.get(plan.provenance, plan.provenance)
    return f"View: camera touchline at bottom · {detail}"


def directions_from_display_plan(
    plan: DisplayOrientationPlan,
    period: int,
    team_names_by_id: dict[int, str],
    shot_based: DirectionInference,
) -> DirectionInference:
    """Attacking directions as an exact consequence of the display transform.

    Raw StatsBomb frames always have the event's own team attacking toward
    increasing x, so a team left unrotated attacks +x on screen and a
    rotated team attacks -x; no statistical estimate is needed. The older
    shot-origin estimate (`shot_based`) is kept as an independent
    cross-check and any disagreement is stated in the explanation, never
    silently resolved. Shared-frame periods (penalty shootout) get no
    arrows: both teams kick at the same goal.
    """
    by_id = plan.attacking_directions(period)
    provenance = orientation_provenance_text(plan, period)
    if by_id is None:
        return DirectionInference({}, "Direction omitted: shared-frame period (both teams attack one goal).", provenance)
    directions = {team_names_by_id[team_id]: direction for team_id, direction in by_id.items()}
    if shot_based.directions and shot_based.directions != directions:
        note = f" WARNING: shot-origin cross-check disagrees ({shot_based.directions})."
    elif shot_based.directions:
        note = " Shot-origin cross-check agrees."
    else:
        note = " Shot-origin cross-check unavailable."
    return DirectionInference(
        directions,
        f"Derived from the display transform for period {period} ({plan.transforms[period].source}).{note}",
        provenance,
    )


def summarize_physical_team_normalization(
    match: MatchFrames, reference_by_period: dict[int, int]
) -> dict[str, int]:
    """Like summarize_team_normalization, but using a period -> reference_team_id
    mapping instead of one fixed id, matching
    apply_team_normalization_to_match_with_physical_calibration.
    """
    counts = {TEAM_NORMALIZATION_REFERENCE: 0, TEAM_NORMALIZATION_MIRRORED: 0}
    for joined in match.usable_frames:
        team = joined.event.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        period = integer_event_field(joined.event, "period")
        key = (
            TEAM_NORMALIZATION_REFERENCE
            if team_id == reference_by_period[period]
            else TEAM_NORMALIZATION_MIRRORED
        )
        counts[key] += 1
    return counts


def summarize_team_normalization(match: MatchFrames, reference_team_id: int) -> dict[str, int]:
    """Count how many usable frames belong to the reference vs. mirrored team."""
    counts = {TEAM_NORMALIZATION_REFERENCE: 0, TEAM_NORMALIZATION_MIRRORED: 0}
    for joined in match.usable_frames:
        team = joined.event.get("team")
        team_id = team.get("id") if isinstance(team, dict) else None
        key = TEAM_NORMALIZATION_REFERENCE if team_id == reference_team_id else TEAM_NORMALIZATION_MIRRORED
        counts[key] += 1
    return counts


def compute_shared_group_info(match: MatchFrames) -> dict[str, SharedGroupInfo]:
    """Detect shared cross-team 360 snapshots (Problem C) on the RAW match.

    Must run on `match` BEFORE Problem A or Problem B correction is applied
    -- see src/team_identity.py. Returns a mapping from event_id to
    SharedGroupInfo for every event whose raw snapshot is shared with at
    least one other team's event; events not part of such a group are
    simply absent from the result (callers should treat a missing key as
    "not shared").
    """
    events = [joined.event for joined in match.usable_frames]
    frames_by_event_id = {joined.event["id"]: joined.frame_360 for joined in match.usable_frames}
    return detect_shared_cross_team_snapshots(events, frames_by_event_id)


# Display-only refinement of Problem C (team_identity.py is shared with the
# dataset builder and is deliberately not changed).
IDENTITY_CONTINUITY_MAX_SECONDS = 2.0
IDENTITY_CONTINUITY_MAX_PAIR_DISTANCE = 3.0
IDENTITY_CONTINUITY_MIN_MISMATCHES = 4
IDENTITY_CONTINUITY_MIN_SHARE = 0.5
IDENTITY_CONTINUITY_MAX_ALTERNATIVE_SHARE = 0.1


def _paired_team_mismatches(
    points_a: list[tuple[tuple[float, float], str]], points_b: list[tuple[tuple[float, float], str]]
) -> tuple[int, int]:
    """Pair two drawn snapshots by optimal assignment on distance (Hungarian;
    greedy nearest-first mis-pairs players in crowded goal-mouths).

    Returns (pairs, pairs whose drawn team differs); pairs further apart than
    IDENTITY_CONTINUITY_MAX_PAIR_DISTANCE and points of unclear team are ignored.
    """
    if not points_a or not points_b:
        return 0, 0
    cost = [[math.dist(a[0], b[0]) for b in points_b] for a in points_a]
    rows, cols = linear_sum_assignment(cost)
    pairs = mismatches = 0
    for i, j in zip(rows, cols):
        if cost[i][j] > IDENTITY_CONTINUITY_MAX_PAIR_DISTANCE:
            continue
        team_a, team_b = points_a[i][1], points_b[j][1]
        if TEAM_IDENTITY_AMBIGUOUS in (team_a, team_b):
            continue
        pairs += 1
        mismatches += team_a != team_b
    return pairs, mismatches


def refine_shared_sources_by_continuity(
    displayed: MatchFrames,
    shared_group_by_event_id: dict[str, SharedGroupInfo],
    raw_events_by_id: dict[str, dict[str, Any]],
    team_names: tuple[str, str],
) -> tuple[dict[str, SharedGroupInfo], list[dict[str, Any]]]:
    """Re-check, for DISPLAY only, which event of a shared cross-team snapshot
    its `teammate` flags belong to.

    Problem C trusts the Problem-A-confident member. The 2026-10-24 colour-
    continuity audit found shared snapshots (a failed Ball Receipt* sharing
    with the other team's Block/Interception) whose colours, read that way,
    contradict both neighbouring frames, while the other member's reading
    makes them continuous. Rule, per shared group: take the nearest drawn
    snapshot before and after it (same period, <= IDENTITY_CONTINUITY_MAX_SECONDS,
    not itself part of a shared group, so its identity is unambiguous). The
    source switches to a member of the other team only if BOTH neighbours
    exist, each disagrees with the current colours on at least
    IDENTITY_CONTINUITY_MIN_MISMATCHES players and IDENTITY_CONTINUITY_MIN_SHARE
    of paired players, and the alternative disagrees on at most
    IDENTITY_CONTINUITY_MAX_ALTERNATIVE_SHARE. Nothing else is recoloured;
    `displayed` must be the match as drawn (after orientation and display plan).
    Returns the refined mapping and one record per switched group.
    """
    frames = displayed.usable_frames
    times = [timestamp_seconds(j.event.get("timestamp")) for j in frames]
    index_by_id = {j.event["id"]: i for i, j in enumerate(frames)}

    def drawn(index: int, info: SharedGroupInfo | None) -> list[tuple[tuple[float, float], str]]:
        event = frames[index].event
        return [
            (tuple(player["location"]), resolve_absolute_team_for_point(player, event, info, raw_events_by_id, team_names))
            for player in frames[index].frame_360.get("freeze_frame") or []
            if isinstance(player.get("location"), list)
        ]

    def neighbour(index: int, step: int) -> int | None:
        period = frames[index].event.get("period")
        k = index + step
        while 0 <= k < len(frames) and abs(times[k] - times[index]) <= IDENTITY_CONTINUITY_MAX_SECONDS:
            if frames[k].event.get("period") != period:
                return None
            if frames[k].event["id"] not in shared_group_by_event_id:
                return k
            k += step
        return None

    refined = dict(shared_group_by_event_id)
    switched: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for info in shared_group_by_event_id.values():
        if info.group_event_ids in seen or info.source_event_id is None:
            continue
        seen.add(info.group_event_ids)
        members = [index_by_id[e] for e in info.group_event_ids if e in index_by_id]
        if not members:
            continue
        current_team = raw_events_by_id[info.source_event_id]["team"]["name"]
        alternative_id = next(
            (e for e in info.group_event_ids if raw_events_by_id[e]["team"]["name"] != current_team), None
        )
        if alternative_id is None:
            continue
        alternative = SharedGroupInfo(info.group_event_ids, alternative_id)
        first, last = min(members), max(members)
        before, after = neighbour(first, -1), neighbour(last, 1)
        if before is None or after is None:
            continue
        evidence = []
        for k in (before, after):
            reference = drawn(k, None)
            current = _paired_team_mismatches(drawn(first, info), reference)
            other = _paired_team_mismatches(drawn(first, alternative), reference)
            evidence.append((current, other))
        if all(
            cur[1] >= IDENTITY_CONTINUITY_MIN_MISMATCHES
            and cur[1] >= IDENTITY_CONTINUITY_MIN_SHARE * cur[0]
            and alt[1] <= IDENTITY_CONTINUITY_MAX_ALTERNATIVE_SHARE * max(alt[0], 1)
            for cur, alt in evidence
        ):
            for event_id in info.group_event_ids:
                refined[event_id] = alternative
            switched.append({
                "events": info.group_event_ids,
                "period": frames[first].event.get("period"),
                "timestamp": frames[first].event.get("timestamp"),
                "previous_source_team": current_team,
                "new_source_team": raw_events_by_id[alternative_id]["team"]["name"],
                "mismatches_current_vs_before_after": [cur for cur, _ in evidence],
                "mismatches_alternative_vs_before_after": [alt for _, alt in evidence],
            })
    return refined, switched


def summarize_team_identity_points(
    frames: list[JoinedFrame],
    shared_group_by_event_id: dict[str, SharedGroupInfo],
    raw_events_by_id: dict[str, dict[str, Any]],
    team_style: TeamStyle,
) -> dict[str, int]:
    """Count every freeze_frame point across `frames` by resolved absolute
    team (team_one/team_two) or AMBIGUOUS, using the same resolution the
    renderer uses when --apply-team-identity-resolution is set.
    """
    team_names = (team_style.team_one, team_style.team_two)
    counts = {team_style.team_one: 0, team_style.team_two: 0, TEAM_IDENTITY_AMBIGUOUS: 0}
    for joined in frames:
        shared_group_info = shared_group_by_event_id.get(joined.event["id"])
        for player in joined.frame_360.get("freeze_frame", []):
            if not isinstance(player, dict):
                continue
            player_team = resolve_absolute_team_for_point(
                player, joined.event, shared_group_info, raw_events_by_id, team_names
            )
            counts[player_team] = counts.get(player_team, 0) + 1
    return counts


def summarize_team_identity(
    match: MatchFrames, shared_group_by_event_id: dict[str, SharedGroupInfo]
) -> dict[str, int]:
    """Report Problem C coverage: how many usable frames are shared, and of
    those, how many have a confidently-identified source vs. remain
    ambiguous.
    """
    total_frames = len(match.usable_frames)
    shared_frames = 0
    recoverable_frames = 0
    ambiguous_frames = 0
    for joined in match.usable_frames:
        info = shared_group_by_event_id.get(joined.event["id"])
        if info is None:
            continue
        shared_frames += 1
        if info.source_event_id is not None:
            recoverable_frames += 1
        else:
            ambiguous_frames += 1
    return {
        "total_frames": total_frames,
        "shared_frames": shared_frames,
        "recoverable_frames": recoverable_frames,
        "ambiguous_frames": ambiguous_frames,
    }


def integer_event_field(event: dict[str, Any], field: str) -> int:
    """Return a required integer event field with a clear validation error."""
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VisualizationError(
            f"Invalid event {event.get('id', '<unknown>')}: {field} must be an integer."
        )
    return value


def chronological_key(joined: JoinedFrame) -> tuple[int, int, int, int]:
    """Sort by period, minute, second, then event index."""
    event = joined.event
    return (
        integer_event_field(event, "period"),
        integer_event_field(event, "minute"),
        integer_event_field(event, "second"),
        integer_event_field(event, "index"),
    )


def parse_period_time(value: str) -> float:
    """Parse period-relative MM:SS or HH:MM:SS text into seconds."""
    parts = value.split(":")
    if len(parts) not in (2, 3) or any(not part.isdigit() for part in parts):
        raise VisualizationError(
            f"Invalid time {value!r}: expected MM:SS or HH:MM:SS."
        )
    numbers = [int(part) for part in parts]
    if len(numbers) == 2:
        minutes, seconds = numbers
        hours = 0
    else:
        hours, minutes, seconds = numbers
        if minutes >= 60:
            raise VisualizationError(f"Invalid time {value!r}: minutes must be below 60.")
    if seconds >= 60:
        raise VisualizationError(f"Invalid time {value!r}: seconds must be below 60.")
    return float(hours * 3600 + minutes * 60 + seconds)


def format_period_time(seconds: float) -> str:
    """Format period-relative seconds for concise selection reporting."""
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, second = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{second:02d}"
    return f"{minutes:02d}:{second:02d}"


def resolve_interval_seconds(
    start_minute: int,
    end_minute: int,
    start_second: float | None = None,
    end_second: float | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> tuple[float, float]:
    """Resolve paired time, second, or minute arguments by documented priority."""
    if (start_time is None) != (end_time is None):
        raise VisualizationError(
            "Invalid time interval: --start-time and --end-time must be provided together."
        )
    if start_time is not None and end_time is not None:
        start_seconds = parse_period_time(start_time)
        end_seconds = parse_period_time(end_time)
        unit = "time"
    else:
        if (start_second is None) != (end_second is None):
            raise VisualizationError(
                "Invalid time interval: --start-second and --end-second must be "
                "provided together."
            )
        if start_second is not None and end_second is not None:
            start_seconds = start_second
            end_seconds = end_second
            unit = "second"
        else:
            start_seconds = float(start_minute * 60)
            end_seconds = float(end_minute * 60)
            unit = "minute"

    if start_seconds < 0:
        raise VisualizationError(f"Invalid time interval: start {unit} must be non-negative.")
    if end_seconds <= start_seconds:
        raise VisualizationError(
            f"Invalid time interval: end {unit} must be greater than start {unit}."
        )
    return start_seconds, end_seconds


def filter_match_interval(
    frames: list[JoinedFrame],
    period: int,
    start_minute: int,
    end_minute: int,
    start_second: float | None = None,
    end_second: float | None = None,
) -> list[JoinedFrame]:
    """Select actual events in a precise half-open period-time interval."""
    if period < 1:
        raise VisualizationError("Invalid period: --period must be at least 1.")
    start_seconds, end_seconds = resolve_interval_seconds(
        start_minute, end_minute, start_second, end_second
    )

    selected = []
    for joined in frames:
        event = joined.event
        if integer_event_field(event, "period") != period:
            continue
        event_seconds = timestamp_seconds(event.get("timestamp"))
        if start_seconds <= event_seconds < end_seconds:
            selected.append(joined)

    if not selected:
        raise VisualizationError(
            "No events in the selected time interval: "
            f"period {period}, seconds [{start_seconds:g}, {end_seconds:g})."
        )
    return selected


def select_actual_event_frames(
    frames: list[JoinedFrame],
    period: int,
    start_minute: int,
    end_minute: int,
    start_second: float | None,
    end_second: float | None,
    start_time: str | None,
    end_time: str | None,
    all_events: bool,
    all_period_events: bool,
    maximum_render_frames: int,
) -> EventSelection:
    """Select real joined events without sampling, duplication, or interpolation."""
    has_explicit_time = any(
        value is not None for value in (start_second, end_second, start_time, end_time)
    )
    if all_events and all_period_events:
        raise VisualizationError("--all-events and --all-period-events are incompatible.")
    if (all_events or all_period_events) and has_explicit_time:
        raise VisualizationError(
            "All-event selection cannot be combined with explicit time arguments."
        )
    if maximum_render_frames < 0:
        raise VisualizationError("--maximum-render-frames must be non-negative.")

    if all_events:
        selected = list(frames)
        selection = EventSelection(
            selected, "all-events", None, None, None, "all periods / all events"
        )
    elif all_period_events:
        selected = [
            joined
            for joined in frames
            if integer_event_field(joined.event, "period") == period
        ]
        selection = EventSelection(
            selected,
            "all-period-events",
            period,
            None,
            None,
            f"period {period} / all events",
        )
    else:
        start, end = resolve_interval_seconds(
            start_minute,
            end_minute,
            start_second,
            end_second,
            start_time,
            end_time,
        )
        selected = [
            joined
            for joined in frames
            if integer_event_field(joined.event, "period") == period
            and start <= timestamp_seconds(joined.event.get("timestamp")) < end
        ]
        selection = EventSelection(
            selected,
            "interval",
            period,
            start,
            end,
            f"{format_period_time(start)}–{format_period_time(end)}",
        )

    if not selection.frames:
        raise VisualizationError(
            f"No usable real StatsBomb 360 events in selection: {selection.interval_label}."
        )
    event_ids = [str(joined.event.get("id")) for joined in selection.frames]
    if len(event_ids) != len(set(event_ids)):
        raise VisualizationError("Selected event UUIDs are not unique; rendering stopped.")
    if maximum_render_frames and len(selection.frames) > maximum_render_frames:
        raise VisualizationError(
            f"Selected {len(selection.frames)} real event frames, exceeding "
            f"--maximum-render-frames {maximum_render_frames}; no frames were truncated."
        )
    return selection


def selection_complexity(frame_count: int) -> str:
    """Classify expected render complexity without changing selection."""
    if frame_count < 100:
        return "short"
    if frame_count < 500:
        return "moderate"
    return "large"


def selection_duration_seconds(selection: EventSelection) -> float:
    """Return a reporting span without creating target timestamps or frames."""
    if selection.interval_start is not None and selection.interval_end is not None:
        return selection.interval_end - selection.interval_start
    spans: list[float] = []
    periods = sorted(
        {integer_event_field(joined.event, "period") for joined in selection.frames}
    )
    for period in periods:
        times = [
            timestamp_seconds(joined.event.get("timestamp"))
            for joined in selection.frames
            if integer_event_field(joined.event, "period") == period
        ]
        if times:
            spans.append(max(times) - min(times))
    return max(sum(spans), 1e-9)


def short_team_label(team_name: str) -> str:
    """Create a readable 2–4 character visual abbreviation."""
    known = TEAM_DISPLAY_LABELS.get(team_name)
    if known is not None:
        return known
    cleaned = team_name.replace("Women's", "").replace("Men's", "").strip()
    words = [word for word in cleaned.replace("-", " ").split() if word]
    if len(words) >= 2:
        return "".join(word[0] for word in words[:4]).upper()
    compact = "".join(character for character in cleaned if character.isalnum())
    return compact[:3].upper() if compact else "TEAM"


def distinct_team_labels(
    team_one: str,
    team_two: str,
    override_one: str | None,
    override_two: str | None,
) -> tuple[str, str]:
    """Resolve legend labels and avoid collisions between two teams."""
    label_one = override_one or short_team_label(team_one)
    label_two = override_two or short_team_label(team_two)
    if label_one.casefold() == label_two.casefold():
        if override_one is not None or override_two is not None:
            raise VisualizationError("Team legend labels must be distinct.")
        label_one = "".join(character for character in team_one if character.isalnum())[:4].upper()
        label_two = "".join(character for character in team_two if character.isalnum())[:4].upper()
        if label_one == label_two:
            label_one, label_two = "T1", "T2"
    return label_one, label_two


def determine_team_style(
    frames: list[JoinedFrame],
    team_one_color: str,
    team_two_color: str,
    team_one_label: str | None = None,
    team_two_label: str | None = None,
) -> TeamStyle:
    """Determine two match teams once and validate their stable colors."""
    team_names: list[str] = []
    for joined in frames:
        team_name = nested_name(joined.event, "team")
        if team_name != "Unknown" and team_name not in team_names:
            team_names.append(team_name)
    if len(team_names) != 2:
        raise VisualizationError(
            f"Expected exactly two match teams, found {len(team_names)}: {team_names}."
        )
    for label, color in (
        ("--team-one-color", team_one_color),
        ("--team-two-color", team_two_color),
    ):
        try:
            matplotlib_colors.to_rgba(color)
        except ValueError as error:
            raise VisualizationError(f"Invalid color for {label}: {color!r}.") from error
    resolved_one_label, resolved_two_label = distinct_team_labels(
        team_names[0], team_names[1], team_one_label, team_two_label
    )
    return TeamStyle(
        team_names[0],
        team_names[1],
        team_one_color,
        team_two_color,
        resolved_one_label,
        resolved_two_label,
    )


def infer_attacking_directions(
    frames: list[JoinedFrame], period: int, team_style: TeamStyle
) -> DirectionInference:
    """Infer directions only when shot origins identify opposite physical ends.

    StatsBomb event coordinates are commonly normalized so both teams attack toward
    increasing x. If both teams' mean shot x is on the same half, those coordinates
    cannot establish stable physical goal sides and no arrows or goal labels are shown.
    """
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for team_name in (team_style.team_one, team_style.team_two):
        shot_x: list[float] = []
        for joined in frames:
            event = joined.event
            if (
                integer_event_field(event, "period") != period
                or nested_name(event, "team") != team_name
                or nested_name(event, "type") != "Shot"
            ):
                continue
            location = coordinate_pair(event.get("location"))
            if location is not None:
                shot_x.append(location[0])
        counts[team_name] = len(shot_x)
        if shot_x:
            means[team_name] = statistics.fmean(shot_x)

    if len(means) == 2:
        first_mean = means[team_style.team_one]
        second_mean = means[team_style.team_two]
        opposite_halves = (first_mean < 60 < second_mean) or (
            second_mean < 60 < first_mean
        )
        if opposite_halves and all(counts[team] >= 2 for team in counts):
            directions = {
                team: (1 if mean_x > 60 else -1) for team, mean_x in means.items()
            }
            return DirectionInference(
                directions,
                "Inferred from each team's mean period shot-origin x; means occupy "
                f"opposite pitch halves ({team_style.team_one}: {first_mean:.1f}, "
                f"{team_style.team_two}: {second_mean:.1f}).",
            )
        return DirectionInference(
            {},
            "Direction omitted: period shot-origin means do not identify opposite "
            "physical ends, consistent with direction-normalized StatsBomb event "
            f"coordinates ({team_style.team_one}: {first_mean:.1f}, "
            f"{team_style.team_two}: {second_mean:.1f}).",
        )
    return DirectionInference(
        {},
        "Direction omitted: both teams need period shot events to infer opposite "
        f"physical ends reliably (shot counts: {counts}).",
    )


def coordinate_pair(value: Any) -> tuple[float, float] | None:
    """Return a numeric coordinate pair, or None when it is unavailable."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    x, y = value
    if isinstance(x, bool) or isinstance(y, bool):
        return None
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return None
    return float(x), float(y)


def visible_area_points(value: Any) -> list[tuple[float, float]] | None:
    """Interpret a flat visible_area sequence as ordered polygon vertices."""
    if not isinstance(value, list) or len(value) < 6 or len(value) % 2:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return [(float(value[i]), float(value[i + 1])) for i in range(0, len(value), 2)]


def nested_name(record: dict[str, Any], field: str) -> str:
    """Read an optional StatsBomb nested object's name."""
    value = record.get(field)
    if isinstance(value, dict) and isinstance(value.get("name"), str):
        return value["name"]
    return "Unknown"


def draw_pitch(ax: Axes, theme: VisualTheme) -> None:
    """Draw unchanged StatsBomb pitch geometry using presentation theme colors."""
    line_color = theme.pitch_lines
    line_width = 1.35
    ax.set_facecolor(theme.pitch_background)
    for zone in range(6):
        ax.add_patch(
            Rectangle(
                (zone * 20, 0),
                20,
                80,
                facecolor=theme.stripe_colors[zone % 2],
                edgecolor="none",
                zorder=0,
            )
        )
    ax.add_patch(
        Rectangle((0, 0), 120, 80, fill=False, ec=line_color, lw=1.8, zorder=1)
    )
    ax.plot([60, 60], [0, 80], color=line_color, lw=line_width, zorder=1)
    ax.add_patch(
        Circle((60, 40), 10, fill=False, ec=line_color, lw=line_width, zorder=1)
    )
    ax.scatter([60], [40], c=line_color, s=11, zorder=1)
    for rectangle in (
        Rectangle((0, 18), 18, 44, fill=False),
        Rectangle((102, 18), 18, 44, fill=False),
        Rectangle((0, 30), 6, 20, fill=False),
        Rectangle((114, 30), 6, 20, fill=False),
    ):
        rectangle.set_edgecolor(line_color)
        rectangle.set_linewidth(line_width)
        rectangle.set_zorder(1)
        ax.add_patch(rectangle)
    ax.scatter([12, 108], [40, 40], c=line_color, s=10, zorder=1)
    # Goals remain outside the unchanged goal lines at x=0 and x=120.
    ax.add_patch(Rectangle((-2, 36), 2, 8, fill=False, ec=line_color, lw=1.5))
    ax.add_patch(Rectangle((120, 36), 2, 8, fill=False, ec=line_color, lw=1.5))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def normalize_view(view: str) -> str:
    """Return the canonical V3 view name, accepting V2 aliases."""
    try:
        return VIEW_ALIASES[view]
    except KeyError as error:
        raise VisualizationError(f"Unsupported view: {view}.") from error


def set_viewport(ax: Axes) -> None:
    """Apply one fixed full-pitch viewport to every event frame."""
    ax.set_xlim(*FIXED_X_LIMITS)
    ax.set_ylim(*FIXED_Y_LIMITS)


# The nominal (pre-stoppage-time) end minute of each period, per the Laws
# of the Game: first half (1) and first half of extra time (3) run 45
# minutes; second half (2) and second half of extra time (4) each run a
# further 45. Period 5 (penalty shootout) has no such nominal duration --
# real broadcasts do not show "120+N" stoppage notation for shootouts, so
# it is deliberately excluded here (see format_match_clock).
PERIOD_NOMINAL_END_MINUTE = {1: 45, 2: 90, 3: 105, 4: 120}

PERIOD_DISPLAY_NAMES = {
    1: "1st Half",
    2: "2nd Half",
    3: "Extra Time, 1st Half",
    4: "Extra Time, 2nd Half",
    5: "Penalty Shootout",
}


def period_display_name(period: Any) -> str:
    """Human-readable period label, or a generic fallback for an
    unexpected period number (never raises)."""
    if isinstance(period, bool) or not isinstance(period, int):
        return "Unknown Period"
    return PERIOD_DISPLAY_NAMES.get(period, f"Period {period}")


def format_match_clock(event: dict[str, Any]) -> str:
    """Format the event's true match-time clock in CONVENTIONAL FOOTBALL
    NOTATION -- e.g. "65:59", "45+2:57", "90+1:04" -- never HH:MM:SS.

    Football match clocks are never hour-wrapped: a 65th-minute event is
    always shown as "65:59", not "1:05:59". The previous implementation
    incorrectly used `divmod(minute, 60)` to introduce an hours component
    once minute exceeded 59 -- a basic misunderstanding of football
    notation, not a football convention at all (confirmed as the exact
    cause of the "1:05:59.487" artifact reported for match 3835342,
    minute=65 -- see
    docs/full_visualization_orientation_and_clock_audit_2026-10-22.md).
    Fixed here: the raw `minute` value is used directly, never wrapped.

    Stoppage time is shown as "<nominal_end>+<overage>:<second>" once
    `minute` strictly exceeds that PERIOD's own nominal end minute (see
    PERIOD_NOMINAL_END_MINUTE) -- keyed off the event's actual `period`
    field, not inferred from the raw minute value alone, so a normal
    second-half minute (e.g. 46) is never misread as first-half stoppage
    time (period 1's nominal end) merely because it is numerically close;
    period 2's own nominal end (90) is what minute=46 is compared against,
    and 46 is far below it, so it renders as plain "46:XX".

    StatsBomb's minute/second fields are the authoritative, continuous
    match clock, already correctly accounting for added time and period
    boundaries (confirmed empirically: period 2 of match 3930160 begins
    at minute=45 even though period 1 ran to minute=47 after 2:57 of
    first-half stoppage; a sampled shootout period resumes at minute=120).
    This function reads them directly -- it never reconstructs elapsed
    time via period-relative-timestamp arithmetic (e.g. "+45"), which
    would be wrong whenever non-zero stoppage time was played.

    The sub-second fractional part is borrowed from the period-relative
    `timestamp` field (minute/second are whole seconds only), since both
    describe the exact same instant.
    """
    minute = event.get("minute")
    second = event.get("second")
    period = event.get("period")
    if (
        isinstance(minute, bool) or not isinstance(minute, int)
        or isinstance(second, bool) or not isinstance(second, int)
    ):
        return "Unknown"
    raw_timestamp = str(event.get("timestamp", ""))
    fractional = "." + raw_timestamp.split(".", 1)[1] if "." in raw_timestamp else ""

    nominal_end = PERIOD_NOMINAL_END_MINUTE.get(period) if isinstance(period, int) and not isinstance(period, bool) else None
    if nominal_end is not None and minute > nominal_end:
        stoppage = minute - nominal_end
        return f"{nominal_end}+{stoppage}:{second:02d}{fractional}"
    return f"{minute:02d}:{second:02d}{fractional}"


def event_header_lines(
    joined: JoinedFrame,
    frame_number: int,
    total_frames: int,
    timing: FrameTiming,
    timing_mode: str,
    show_timing_debug: bool = False,
) -> tuple[str, str]:
    """Build compact primary and secondary presentation header lines.

    Primary line uses conventional FOOTBALL match-clock notation (see
    format_match_clock -- e.g. "65:59" or "45+2:57", never HH:MM:SS),
    led by the human-readable period name (e.g. "2nd Half"), matching how
    a football viewer actually reads a match clock -- NOT how a general-
    purpose timer/stopwatch would be formatted. The period-relative
    StatsBomb `timestamp` field (resets to 00:00 at the start of each
    period) is technical/debugging information only and is demoted to the
    secondary line.
    """
    event = joined.event
    match_clock = format_match_clock(event)
    period_name = period_display_name(event.get("period"))
    period_relative = str(event.get("timestamp", "Unknown"))
    if period_relative.startswith("00:"):
        period_relative = period_relative[3:]
    previous = (
        "—"
        if timing.gap_from_previous is None
        else f"{timing.gap_from_previous:.3f} s"
    )
    following = (
        "—" if timing.gap_to_next is None else f"{timing.gap_to_next:.3f} s"
    )
    player = nested_name(event, "player")
    primary_parts = [period_name, match_clock, nested_name(event, "type")]
    if player != "Unknown":
        primary_parts.append(player)
    primary = "  ·  ".join(primary_parts)
    next_text = "next —" if following == "—" else f"next frame in {following}"
    secondary = (
        f"Period-relative {period_relative}  |  "
        f"Frame {frame_number}/{total_frames}  |  previous {previous}  |  "
        f"{next_text}"
    )
    if show_timing_debug:
        secondary += (
            f"  |  timing {timing_mode}  |  "
            f"GIF hold {timing.display_duration_ms} ms"
        )
    return primary, secondary


def add_orientation_annotations(
    ax: Axes, team_style: TeamStyle, direction_inference: DirectionInference
) -> None:
    """Show goal-side labels and arrows only for reliably inferred directions.

    Label text is deliberately unambiguous ("{team} defending goal", never
    bare "{team} goal") -- confirmed, 2026-10-22, to be a real source of
    viewer misreading (see
    docs/russia_belgium_orientation_regression_2026-10-22.md): "{team}
    goal" alone is genuinely ambiguous in English (could mean the goal a
    team defends OR the goal a team scores into), and the label has always
    placed the text at the team's DEFENDED end (`goal_x = 0 if direction
    == 1 else 120` -- direction 1 means the team attacks toward
    increasing x, so their own goal is at the opposite, x=0, end). This
    is a text-only clarification: neither `goal_x` nor the arrow geometry
    changes, so no match's rendered COORDINATES are affected by this fix.
    """
    if direction_inference.provenance:
        # Bottom margin, i.e. beside the camera (near) touchline.
        ax.text(
            120,
            81.7,
            direction_inference.provenance,
            color="#9aa0a6",
            fontsize=7.4,
            ha="right",
            va="center",
            clip_on=False,
        )
    for team_name, direction in direction_inference.directions.items():
        color = team_style.color_for(team_name)
        goal_x = 0 if direction == 1 else 120
        ax.text(
            goal_x,
            GOAL_LABEL_Y,
            f"{team_name} defending goal",
            color=color,
            fontsize=9,
            fontweight="bold",
            ha="left" if goal_x == 0 else "right",
            va="center",
            clip_on=False,
        )
        arrow_start = 44 if direction == 1 else 76
        ax.annotate(
            "attacks",
            xy=(arrow_start + direction * 16, GOAL_LABEL_Y),
            xytext=(arrow_start, GOAL_LABEL_Y),
            color=color,
            fontsize=8,
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.5},
            annotation_clip=False,
        )


def render_event_frame(
    ax: Axes,
    match_id: int,
    joined: JoinedFrame,
    frame_number: int,
    total_frames: int,
    timing: FrameTiming,
    timing_mode: str,
    view: str,
    team_style: TeamStyle,
    direction_inference: DirectionInference,
    theme: VisualTheme,
    show_timing_debug: bool = False,
    highlight_goalkeepers: bool = False,
    shared_group_by_event_id: dict[str, SharedGroupInfo] | None = None,
    raw_events_by_id: dict[str, dict[str, Any]] | None = None,
    frame_identities: dict[str, FrameIdentity] | None = None,
) -> None:
    """Render one actual StatsBomb 360 event freeze frame.

    `frame_identities` (src/team_identity.py resolve_match_team_identity,
    shared with the dataset builder) takes precedence over the Problem C
    inputs below when given: each point is drawn in its resolved team's
    colour, points corrected on evidence get a dashed ring, and UNRESOLVED
    points the neutral "team unclear" fill -- only those points, never the
    whole frame unless every point of it is unresolved.

    `shared_group_by_event_id`/`raw_events_by_id` are Problem C's inputs
    (see src/team_identity.py): when provided, each non-actor point's
    absolute team is resolved via `resolve_absolute_team_for_point` instead
    of the plain teammate-relative-to-this-event logic, which is only
    reliable when the frame's raw 360 snapshot isn't shared with a
    different team's event. Coordinates are never touched by this; it only
    changes which color a point is drawn with.
    """
    ax.clear()
    draw_pitch(ax, theme)
    frame = joined.frame_360
    polygon_points = visible_area_points(frame.get("visible_area"))
    set_viewport(ax)

    if view == "camera-overlay" and polygon_points is not None:
        ax.add_patch(
            Polygon(
                polygon_points,
                closed=True,
                facecolor=theme.camera_fill,
                edgecolor=theme.camera_edge,
                alpha=theme.camera_alpha,
                lw=1.25,
                zorder=2,
            )
        )

    team_names = (team_style.team_one, team_style.team_two)
    shared_group_info = (
        shared_group_by_event_id.get(joined.event["id"]) if shared_group_by_event_id is not None else None
    )
    player_groups: dict[tuple[str, bool, bool], list[tuple[float, float]]] = {}
    corrected_locations: list[tuple[float, float]] = []
    has_ambiguous_point = False
    identity = frame_identities.get(joined.event["id"]) if frame_identities is not None else None

    # Freeze-frame points lack reliable IDs for non-actors, so no identities,
    # labels, trails, nearest-neighbour matching, or tracking are invented.
    for point_index, player in enumerate(frame["freeze_frame"]):
        if not isinstance(player, dict):
            continue
        location = coordinate_pair(player.get("location"))
        if location is None:
            continue
        if identity is not None:
            player_team = identity.teams[point_index]
            if identity.statuses[point_index] in (IDENTITY_CORRECTED_VIEW, IDENTITY_CORRECTED_GOALKEEPER):
                corrected_locations.append(location)
        elif raw_events_by_id is not None:
            player_team = resolve_absolute_team_for_point(
                player, joined.event, shared_group_info, raw_events_by_id, team_names
            )
        else:
            event_team = nested_name(joined.event, "team")
            player_team = event_team if player.get("teammate") is True else team_style.opponent_of(event_team)
        if player_team == TEAM_IDENTITY_AMBIGUOUS:
            has_ambiguous_point = True
        key = (player_team, player.get("keeper") is True, player.get("actor") is True)
        player_groups.setdefault(key, []).append(location)

    for (player_team, is_keeper, is_actor), locations in player_groups.items():
        x, y = zip(*locations)
        color = theme.ambiguous_fill if player_team == TEAM_IDENTITY_AMBIGUOUS else team_style.color_for(player_team)
        ax.scatter(
            x,
            y,
            c=color,
            marker="D" if is_keeper else "o",
            edgecolors=theme.marker_edge,
            linewidths=2.0 if is_keeper else 1.6,
            s=167 if is_keeper else 143,
            zorder=6,
        )
        if is_keeper and highlight_goalkeepers:
            ax.scatter(
                x,
                y,
                facecolors="none",
                edgecolors=theme.primary_text,
                marker="D",
                linewidths=1.7,
                s=250,
                zorder=7,
            )
        if is_actor:
            ax.scatter(
                x,
                y,
                facecolors="none",
                edgecolors=theme.actor_accent,
                marker="o",
                linewidths=3.2,
                s=265,
                zorder=8,
            )
            ax.scatter(
                x,
                y,
                facecolors="none",
                edgecolors=theme.actor_accent,
                marker="o",
                linewidths=1.4,
                s=340,
                alpha=0.28,
                zorder=7,
            )
            ax.scatter(
                x,
                y,
                c=theme.actor_accent,
                marker="o",
                edgecolors="none",
                s=18,
                zorder=9,
            )

    if corrected_locations:
        x, y = zip(*corrected_locations)
        ax.scatter(
            x, y, facecolors="none", edgecolors=theme.correction_ring, marker="o",
            linewidths=1.5, linestyle=(0, (2, 1.5)), s=420, zorder=7,
        )

    ball = coordinate_pair(joined.event.get("location"))
    if ball is not None:
        ax.scatter(
            [ball[0]],
            [ball[1]],
            facecolors="none",
            edgecolors=theme.primary_text,
            linewidths=1.35,
            s=112,
            alpha=0.60,
            zorder=10,
        )
        ax.scatter(
            [ball[0]],
            [ball[1]],
            c=theme.ball_fill,
            edgecolors=theme.ball_edge,
            linewidths=1.8,
            s=72,
            zorder=11,
        )

    add_orientation_annotations(ax, team_style, direction_inference)
    primary_header, secondary_header = event_header_lines(
        joined,
        frame_number,
        total_frames,
        timing,
        timing_mode,
        show_timing_debug,
    )
    ax.text(
        0.0,
        1.055,
        primary_header,
        transform=ax.transAxes,
        color=theme.primary_text,
        fontsize=13.5,
        fontweight="semibold",
        ha="left",
        va="bottom",
        clip_on=False,
    )
    ax.text(
        0.0,
        1.018,
        secondary_header,
        transform=ax.transAxes,
        color=theme.secondary_text,
        fontsize=9.2,
        ha="left",
        va="bottom",
        clip_on=False,
    )
    legend_handles: list[Any] = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=team_style.team_one_color,
            markeredgecolor=theme.marker_edge, markeredgewidth=1.2,
            markersize=8.5, label=team_style.team_one_label,
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=team_style.team_two_color,
            markeredgecolor=theme.marker_edge, markeredgewidth=1.2,
            markersize=8.5, label=team_style.team_two_label,
        ),
        Line2D(
            [0], [0], marker="D", color="none",
            markerfacecolor=theme.secondary_text,
            markeredgecolor=theme.marker_edge, markeredgewidth=1.2,
            markersize=8, label="Goalkeeper",
        ),
        Line2D(
            [0], [0], marker="o", color="none", markerfacecolor="none",
            markeredgecolor=theme.actor_accent, markeredgewidth=2,
            markersize=10, label="Event actor",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor=theme.ball_fill, markeredgecolor=theme.ball_edge,
            markeredgewidth=1.2, markersize=6.5,
            label="Ball / event location",
        ),
    ]
    if view == "camera-overlay":
        legend_handles.append(
            Patch(
                facecolor=theme.camera_fill,
                edgecolor=theme.camera_edge,
                alpha=max(theme.camera_alpha, 0.35),
                label="Visible camera area",
            )
        )
    if has_ambiguous_point:
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=theme.ambiguous_fill,
                markeredgecolor=theme.marker_edge, markeredgewidth=1.2,
                markersize=8.5,
                label="Team unclear (identity not established)" if identity is not None
                else "Team unclear (shared 360 snapshot)",
            )
        )
    if corrected_locations:
        legend_handles.append(
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor="none",
                markeredgecolor=theme.correction_ring, markeredgewidth=1.4, linestyle="none",
                markersize=11, label="Team corrected on evidence",
            )
        )
    visible_x = [
        location[0]
        for locations in player_groups.values()
        for location in locations
    ]
    legend_location = (
        2 if visible_x and statistics.fmean(visible_x) >= 60 else 1
    )
    # Anchored just inside the far touchline (axes fraction of y=0 on the
    # inverted y-axis), so it never covers the goal labels in the margin.
    far_touchline = (FIXED_Y_LIMITS[0] - 0.6) / (FIXED_Y_LIMITS[0] - FIXED_Y_LIMITS[1])
    legend = ax.legend(
        handles=legend_handles,
        loc=legend_location,
        bbox_to_anchor=(0.0, 0.0, 1.0, far_touchline),
        borderaxespad=0.65,
        borderpad=0.55,
        labelspacing=0.38,
        handletextpad=0.5,
        handlelength=1.2,
        fontsize=7.8,
        framealpha=0.88,
        facecolor=theme.legend_face,
        edgecolor=theme.legend_edge,
    )
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color(theme.primary_text)


def timing_mode_slug(timing_mode: str) -> str:
    """Return a filename-safe timing mode."""
    return timing_mode.replace("-", "_")


def default_output_path(
    match_id: int,
    selection: EventSelection,
    theme: str,
    timing_mode: str,
    orientation_corrected: bool = False,
    team_normalized: bool = False,
    team_identity_resolved: bool = False,
) -> Path:
    """Return a generic selection-aware GIF path under results."""
    if selection.mode == "all-events":
        selection_slug = "all_periods_all_events"
    elif selection.mode == "all-period-events":
        selection_slug = f"period_{selection.period}_all_events"
    else:
        start = f"{selection.interval_start:g}".replace(".", "p")
        end = f"{selection.interval_end:g}".replace(".", "p")
        selection_slug = f"period_{selection.period}_seconds_{start}_{end}"
    suffix = (
        ("_oriented" if orientation_corrected else "")
        + ("_teamnorm" if team_normalized else "")
        + ("_teamid" if team_identity_resolved else "")
    )
    return (
        PROJECT_ROOT
        / "results"
        / f"statsbomb_360_match_{match_id}_{selection_slug}_{theme}_"
        f"{timing_mode_slug(timing_mode)}{suffix}.gif"
    )


def static_test_frame_path(output_path: Path) -> Path:
    """Return the PNG test-frame path paired with an animation output path."""
    return output_path.with_name(f"{output_path.stem}_test_frame.png")


def create_landscape_figure(theme: VisualTheme) -> tuple[Any, Axes]:
    """Create a pitch-dominant 16:9 canvas using the selected theme."""
    figure, ax = plt.subplots(figsize=(16, 9), facecolor=theme.figure_background)
    figure.patch.set_facecolor(theme.figure_background)
    figure.subplots_adjust(left=0.015, right=0.985, top=0.92, bottom=0.005)
    return figure, ax


def save_static_test_frame(
    match_id: int,
    selected: list[JoinedFrame],
    output_path: Path,
    view: str,
    team_style: TeamStyle,
    direction_inference: DirectionInference,
    timing_mode: str = "fixed",
    timing_plan: list[FrameTiming] | None = None,
    theme: VisualTheme | None = None,
    show_timing_debug: bool = False,
    highlight_goalkeepers: bool = False,
    shared_group_by_event_id: dict[str, SharedGroupInfo] | None = None,
    raw_events_by_id: dict[str, dict[str, Any]] | None = None,
    frame_identities: dict[str, FrameIdentity] | None = None,
) -> Path:
    """Render and save the first selected event as a structural test image."""
    test_frame_path = static_test_frame_path(output_path)
    theme = theme or get_visual_theme("light")
    if timing_plan is None:
        timing_plan = build_timing_plan(selected, timing_mode, 250, 1.0, 80, 1500)
    try:
        test_frame_path.parent.mkdir(parents=True, exist_ok=True)
        figure, ax = create_landscape_figure(theme)
        render_event_frame(
            ax,
            match_id,
            selected[0],
            1,
            len(selected),
            timing_plan[0],
            timing_mode,
            view,
            team_style,
            direction_inference,
            theme,
            show_timing_debug,
            highlight_goalkeepers,
            shared_group_by_event_id,
            raw_events_by_id,
            frame_identities=frame_identities,
        )
        figure.savefig(
            test_frame_path, dpi=150, facecolor=theme.figure_background
        )
        plt.close(figure)
    except (OSError, ValueError) as error:
        raise VisualizationError(
            f"Invalid output path {test_frame_path}: {error}"
        ) from error
    return test_frame_path


def print_goalkeeper_frames(frames: list[JoinedFrame]) -> int:
    """Print selected raw freeze-frame points explicitly marked keeper=true."""
    matching_frames = 0
    for frame_number, joined in enumerate(frames, start=1):
        keepers = [
            player
            for player in joined.frame_360["freeze_frame"]
            if isinstance(player, dict) and player.get("keeper") is True
        ]
        if not keepers:
            continue
        matching_frames += 1
        event = joined.event
        print(f"Goalkeeper frame {frame_number}/{len(frames)}:")
        print(f"timestamp={event.get('timestamp')}")
        print(f"event_id={event.get('id')}")
        print(f"event_type={nested_name(event, 'type')}")
        print(f"event_team={nested_name(event, 'team')}")
        print(f"keepers={len(keepers)}")
        for keeper_number, keeper in enumerate(keepers, start=1):
            print(
                f"keeper[{keeper_number}]: location={keeper.get('location')}, "
                f"teammate={str(keeper.get('teammate')).lower()}"
            )
    if matching_frames == 0:
        print("No selected freeze frames contain keeper=true points.")
    return matching_frames


def point_inside_pitch(point: tuple[float, float]) -> bool:
    """Return whether a coordinate lies inside the StatsBomb pitch boundary."""
    return 0.0 <= point[0] <= 120.0 and 0.0 <= point[1] <= 80.0


def validate_selected_frames(frames: list[JoinedFrame]) -> StructuralReport:
    """Audit requested coordinate invariants without modifying raw coordinates."""
    player_count = 0
    player_outside = 0
    player_outside_frames: set[str] = set()
    ball_count = 0
    ball_outside = 0
    visible_area_count = 0
    visible_area_odd = 0
    polygon_point_count = 0
    polygon_outside = 0

    for joined in frames:
        event_uuid = str(joined.event.get("id", "<unknown>"))
        for player_number, player in enumerate(
            joined.frame_360["freeze_frame"], start=1
        ):
            if not isinstance(player, dict):
                raise VisualizationError(
                    f"Invalid freeze_frame for event {event_uuid}: player "
                    f"{player_number} is not an object."
                )
            player_point = coordinate_pair(player.get("location"))
            if player_point is None:
                raise VisualizationError(
                    f"Invalid player coordinate for event {event_uuid}, "
                    f"player {player_number}."
                )
            player_count += 1
            if not point_inside_pitch(player_point):
                player_outside += 1
                player_outside_frames.add(event_uuid)

        raw_ball = joined.event.get("location")
        if raw_ball is not None:
            ball_point = coordinate_pair(raw_ball)
            if ball_point is None:
                raise VisualizationError(
                    f"Invalid ball/event coordinate for event {event_uuid}."
                )
            ball_count += 1
            if not point_inside_pitch(ball_point):
                ball_outside += 1

        raw_visible_area = joined.frame_360.get("visible_area")
        if raw_visible_area is not None:
            visible_area_count += 1
        if not isinstance(raw_visible_area, list):
            raise VisualizationError(
                f"Invalid visible_area for event {event_uuid}: expected a list."
            )
        if len(raw_visible_area) % 2:
            visible_area_odd += 1
            continue
        polygon = visible_area_points(raw_visible_area)
        if polygon is None:
            raise VisualizationError(
                f"Invalid visible_area polygon for event {event_uuid}: expected "
                "at least three numeric coordinate pairs."
            )
        polygon_point_count += len(polygon)
        polygon_outside += sum(not point_inside_pitch(point) for point in polygon)

    return StructuralReport(
        player_coordinate_count=player_count,
        player_outside_count=player_outside,
        player_outside_frame_count=len(player_outside_frames),
        ball_coordinate_count=ball_count,
        ball_outside_count=ball_outside,
        visible_area_count=visible_area_count,
        visible_area_odd_count=visible_area_odd,
        polygon_point_count=polygon_point_count,
        polygon_outside_count=polygon_outside,
    )


def timestamp_seconds(timestamp: Any) -> float:
    """Convert a StatsBomb HH:MM:SS.sss timestamp to period-relative seconds."""
    if not isinstance(timestamp, str):
        raise VisualizationError("Invalid event timestamp: expected a string.")
    try:
        parsed = datetime.strptime(timestamp, "%H:%M:%S.%f")
    except ValueError as error:
        raise VisualizationError(f"Invalid event timestamp {timestamp!r}.") from error
    return parsed.hour * 3600 + parsed.minute * 60 + parsed.second + parsed.microsecond / 1e6


def percentile(values: list[float], percentage: float) -> float | None:
    """Calculate a linearly interpolated percentile using the standard library."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentage / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def build_timing_plan(
    frames: list[JoinedFrame],
    timing_mode: str,
    fixed_duration_ms: int,
    playback_speed: float,
    minimum_duration_ms: int,
    maximum_duration_ms: int,
) -> list[FrameTiming]:
    """Plan GIF holds from real event gaps without adding synthetic frames."""
    if fixed_duration_ms <= 0:
        raise VisualizationError(
            "Invalid frame duration: --frame-duration-ms must be greater than 0."
        )
    if playback_speed <= 0:
        raise VisualizationError("Invalid playback speed: must be greater than 0.")
    if minimum_duration_ms <= 0:
        raise VisualizationError("Minimum frame duration must be greater than 0.")
    if maximum_duration_ms < minimum_duration_ms:
        raise VisualizationError(
            "Maximum frame duration must be greater than or equal to the minimum."
        )

    times = [timestamp_seconds(frame.event.get("timestamp")) for frame in frames]
    periods = [integer_event_field(frame.event, "period") for frame in frames]
    gaps: list[float | None] = []
    for index in range(len(times) - 1):
        if periods[index] != periods[index + 1]:
            gaps.append(None)
            continue
        gap = times[index + 1] - times[index]
        if gap < 0:
            raise VisualizationError("Selected event timestamps are not chronological.")
        gaps.append(gap)

    timings = []
    for index, match_second in enumerate(times):
        gap_from = None if index == 0 else gaps[index - 1]
        gap_to = None if index == len(times) - 1 else gaps[index]
        if timing_mode == "fixed":
            display_duration = fixed_duration_ms
        else:
            # The last real frame has no selected successor; use the minimum hold.
            real_duration = 0.0 if gap_to is None else gap_to * 1000 / playback_speed
            display_duration = round(
                min(max(real_duration, minimum_duration_ms), maximum_duration_ms)
            )
        timings.append(
            FrameTiming(match_second, gap_from, gap_to, int(display_duration))
        )
    return timings


def calculate_temporal_density(
    timings: list[FrameTiming], requested_seconds: float
) -> TemporalDensity:
    """Summarize real gaps between selected event-based freeze frames."""
    gaps = [
        timing.gap_to_next
        for timing in timings
        if timing.gap_to_next is not None
    ]
    numeric_gaps = [float(gap) for gap in gaps]
    gap_count = len(numeric_gaps)
    bin_specs = (
        ("below 0.5 seconds", lambda gap: gap < 0.5),
        ("0.5–1 second", lambda gap: 0.5 <= gap < 1.0),
        ("1–2 seconds", lambda gap: 1.0 <= gap < 2.0),
        ("2–5 seconds", lambda gap: 2.0 <= gap <= 5.0),
        ("above 5 seconds", lambda gap: gap > 5.0),
    )
    bins = tuple(
        (
            label,
            count := sum(predicate(gap) for gap in numeric_gaps),
            100 * count / gap_count if gap_count else 0.0,
        )
        for label, predicate in bin_specs
    )
    return TemporalDensity(
        requested_seconds=requested_seconds,
        frame_count=len(timings),
        frames_per_second=len(timings) / requested_seconds,
        average_gap=statistics.fmean(numeric_gaps) if numeric_gaps else None,
        minimum_gap=min(numeric_gaps) if numeric_gaps else None,
        percentile_25=percentile(numeric_gaps, 25),
        median_gap=statistics.median(numeric_gaps) if numeric_gaps else None,
        percentile_75=percentile(numeric_gaps, 75),
        maximum_gap=max(numeric_gaps) if numeric_gaps else None,
        bins=bins,
    )


def export_timing_csv(
    output_path: Path,
    match_id: int,
    period: int,
    frames: list[JoinedFrame],
    timings: list[FrameTiming],
    timing_mode: str,
) -> Path:
    """Export one timing row per actual selected freeze frame."""
    fieldnames = [
        "match_id",
        "period",
        "frame_number",
        "event_id",
        "event_index",
        "timestamp",
        "match_second",
        "event_type",
        "team",
        "player",
        "visible_player_count",
        "gap_from_previous_seconds",
        "gap_to_next_seconds",
        "gif_display_duration_ms",
        "timing_mode",
    ]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for number, (joined, timing) in enumerate(
                zip(frames, timings, strict=True), start=1
            ):
                event = joined.event
                player = nested_name(event, "player")
                writer.writerow(
                    {
                        "match_id": match_id,
                        "period": period,
                        "frame_number": number,
                        "event_id": event.get("id"),
                        "event_index": event.get("index"),
                        "timestamp": event.get("timestamp"),
                        "match_second": f"{timing.match_second:.3f}",
                        "event_type": nested_name(event, "type"),
                        "team": nested_name(event, "team"),
                        "player": "" if player == "Unknown" else player,
                        "visible_player_count": len(
                            joined.frame_360["freeze_frame"]
                        ),
                        "gap_from_previous_seconds": (
                            ""
                            if timing.gap_from_previous is None
                            else f"{timing.gap_from_previous:.3f}"
                        ),
                        "gap_to_next_seconds": (
                            ""
                            if timing.gap_to_next is None
                            else f"{timing.gap_to_next:.3f}"
                        ),
                        "gif_display_duration_ms": timing.display_duration_ms,
                        "timing_mode": timing_mode,
                    }
                )
    except OSError as error:
        raise VisualizationError(
            f"Invalid timing CSV path {output_path}: {error}"
        ) from error
    return output_path


def export_event_frame_manifest(
    output_path: Path,
    match_id: int,
    frames: list[JoinedFrame],
    timings: list[FrameTiming],
) -> Path:
    """Export exactly one manifest row per selected real event frame."""
    fieldnames = [
        "frame_number",
        "match_id",
        "period",
        "event_id",
        "event_index",
        "timestamp",
        "match_time",
        "period_second",
        "event_type",
        "team",
        "player",
        "visible_player_count",
        "goalkeeper_count",
        "actor_count",
        "has_visible_area",
        "gap_from_previous_seconds",
        "gap_to_next_seconds",
        "gif_display_duration_ms",
    ]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for frame_number, (joined, timing) in enumerate(
                zip(frames, timings, strict=True), start=1
            ):
                event = joined.event
                freeze_frame = joined.frame_360["freeze_frame"]
                player = nested_name(event, "player")
                writer.writerow(
                    {
                        "frame_number": frame_number,
                        "match_id": match_id,
                        "period": event.get("period"),
                        "event_id": event.get("id"),
                        "event_index": event.get("index"),
                        "timestamp": event.get("timestamp"),
                        "match_time": format_match_clock(event),
                        "period_second": f"{timing.match_second:.3f}",
                        "event_type": nested_name(event, "type"),
                        "team": nested_name(event, "team"),
                        "player": "" if player == "Unknown" else player,
                        "visible_player_count": len(freeze_frame),
                        "goalkeeper_count": sum(
                            isinstance(point, dict) and point.get("keeper") is True
                            for point in freeze_frame
                        ),
                        "actor_count": sum(
                            isinstance(point, dict) and point.get("actor") is True
                            for point in freeze_frame
                        ),
                        "has_visible_area": (
                            visible_area_points(joined.frame_360.get("visible_area"))
                            is not None
                        ),
                        "gap_from_previous_seconds": (
                            "" if timing.gap_from_previous is None
                            else f"{timing.gap_from_previous:.3f}"
                        ),
                        "gap_to_next_seconds": (
                            "" if timing.gap_to_next is None
                            else f"{timing.gap_to_next:.3f}"
                        ),
                        "gif_display_duration_ms": timing.display_duration_ms,
                    }
                )
    except OSError as error:
        raise VisualizationError(
            f"Invalid event-frame manifest path {output_path}: {error}"
        ) from error
    return output_path


def save_timing_plot(
    output_path: Path,
    timings: list[FrameTiming],
    interval_start: float,
    interval_end: float,
) -> Path:
    """Plot real event times and gaps to subsequent 360 freeze frames."""
    figure, ax = plt.subplots(figsize=(12, 5.5))
    times = [timing.match_second for timing in timings]
    gap_times = [
        timing.match_second for timing in timings if timing.gap_to_next is not None
    ]
    gaps = [timing.gap_to_next for timing in timings if timing.gap_to_next is not None]
    ax.plot(gap_times, gaps, marker="o", ms=4, lw=1.2, color="#2474B5")
    rug_height = max([float(gap) for gap in gaps] + [1.0]) * 0.045
    ax.vlines(times, 0, rug_height, color="#D64545", alpha=0.75, lw=1)
    ax.set_xlim(interval_start, interval_end)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Match time within selected period (seconds)")
    ax.set_ylabel("Gap to next available 360 freeze frame (seconds)")
    ax.set_title(
        "Temporal density of event-based StatsBomb 360 freeze frames\n"
        "Irregular event observations — not regularly sampled tracking frames"
    )
    ax.grid(alpha=0.25)
    figure.tight_layout()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=150)
    except OSError as error:
        raise VisualizationError(
            f"Invalid timing plot path {output_path}: {error}"
        ) from error
    finally:
        plt.close(figure)
    return output_path


def summarize_segment(frames: list[JoinedFrame]) -> SegmentSummary:
    """Compute descriptive statistics over actual consecutive event frames."""
    times = [timestamp_seconds(joined.event.get("timestamp")) for joined in frames]
    periods = [integer_event_field(joined.event, "period") for joined in frames]
    gaps = [
        times[index + 1] - times[index]
        for index in range(len(times) - 1)
        if periods[index] == periods[index + 1]
    ]
    if any(gap < 0 for gap in gaps):
        raise VisualizationError("Selected event timestamps are not chronological.")
    if not gaps:
        gaps = [0.0]

    visible_counts = [len(joined.frame_360["freeze_frame"]) for joined in frames]
    valid_ball_frames = sum(
        coordinate_pair(joined.event.get("location")) is not None for joined in frames
    )
    valid_polygon_frames = sum(
        visible_area_points(joined.frame_360.get("visible_area")) is not None
        for joined in frames
    )
    return SegmentSummary(
        first_timestamp=str(frames[0].event.get("timestamp")),
        last_timestamp=str(frames[-1].event.get("timestamp")),
        average_gap=statistics.fmean(gaps),
        minimum_gap=min(gaps),
        median_gap=statistics.median(gaps),
        maximum_gap=max(gaps),
        average_visible_players=statistics.fmean(visible_counts),
        minimum_visible_players=min(visible_counts),
        maximum_visible_players=max(visible_counts),
        valid_ball_frames=valid_ball_frames,
        valid_polygon_frames=valid_polygon_frames,
    )


def save_animation(
    match_id: int,
    selected: list[JoinedFrame],
    frame_duration_ms: int,
    output_path: Path,
    view: str,
    team_style: TeamStyle,
    direction_inference: DirectionInference,
    timing_mode: str = "fixed",
    timing_plan: list[FrameTiming] | None = None,
    theme: VisualTheme | None = None,
    show_timing_debug: bool = False,
    highlight_goalkeepers: bool = False,
    shared_group_by_event_id: dict[str, SharedGroupInfo] | None = None,
    raw_events_by_id: dict[str, dict[str, Any]] | None = None,
    frame_identities: dict[str, FrameIdentity] | None = None,
) -> Path:
    """Save one animation frame per actual event, without interpolation."""
    if frame_duration_ms <= 0:
        raise VisualizationError(
            "Invalid frame duration: --frame-duration-ms must be greater than 0."
        )
    if output_path.suffix.lower() != ".gif":
        raise VisualizationError(
            f"Invalid output path {output_path}: output must have a .gif extension."
        )

    theme = theme or get_visual_theme("light")
    if timing_plan is None:
        timing_plan = build_timing_plan(
            selected, timing_mode, frame_duration_ms, 1.0, 80, 1500
        )
    if len(timing_plan) != len(selected):
        raise VisualizationError("Timing plan length does not match selected frames.")

    figure, ax = create_landscape_figure(theme)

    def update(frame_index: int) -> tuple[()]:
        timing = timing_plan[frame_index]
        render_event_frame(
            ax,
            match_id,
            selected[frame_index],
            frame_index + 1,
            len(selected),
            timing,
            timing_mode,
            view,
            team_style,
            direction_inference,
            theme,
            show_timing_debug,
            highlight_goalkeepers,
            shared_group_by_event_id,
            raw_events_by_id,
            frame_identities=frame_identities,
        )
        return ()

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if timing_mode == "fixed":
            event_animation = animation.FuncAnimation(
                figure,
                update,
                frames=len(selected),
                interval=frame_duration_ms,
                repeat=False,
                blit=False,
            )
            event_animation.save(
                output_path,
                writer=animation.PillowWriter(
                    fps=1000.0 / frame_duration_ms,  # type: ignore[arg-type]
                    metadata={
                        "comment": (
                            "Event-based StatsBomb 360 freeze frames; no "
                            "interpolation or fixed-time resampling."
                        )
                    },
                ),
                dpi=100,
            )
        else:
            pillow_frames: list[Image.Image] = []
            for frame_index in range(len(selected)):
                update(frame_index)
                figure.canvas.draw()
                buffer = io.BytesIO()
                figure.savefig(
                    buffer,
                    format="png",
                    dpi=100,
                    facecolor=theme.figure_background,
                )
                buffer.seek(0)
                with Image.open(buffer) as rendered:
                    pillow_frames.append(rendered.convert("RGB"))
            durations = [timing.display_duration_ms for timing in timing_plan]
            pillow_frames[0].save(
                output_path,
                save_all=True,
                append_images=pillow_frames[1:],
                duration=durations,
                loop=0,
                optimize=False,
                comment=(
                    b"Event-based StatsBomb 360 freeze frames; variable real-time "
                    b"holds, no interpolation or resampling."
                ),
            )
    except (OSError, ValueError, RuntimeError) as error:
        raise VisualizationError(f"Invalid output path {output_path}: {error}") from error
    finally:
        plt.close(figure)
    return output_path


def print_match_summary(match_id: int, metadata: MatchMetadata | None) -> None:
    """Print concise optional match metadata."""
    print(f"Match {match_id}")
    if metadata is None:
        print("Metadata: unavailable")
        return
    print(f"{metadata.home_team} vs {metadata.away_team}")
    print(f"Competition: {metadata.competition}")
    print(f"Season: {metadata.season}")
    print(f"Date: {metadata.date}")
    print(f"Score: {metadata.score}")


def print_available_matches(
    rows: list[tuple[int, MatchMetadata | None]],
) -> None:
    """Print available 360 matches with metadata and usable record counts."""
    if not rows:
        print("No available StatsBomb 360 matches match the requested filters.")
        return
    headings = (
        "match_id", "date", "competition", "season", "home_team", "away_team",
        "score", "360_records", "usable_freeze_frames",
    )
    print(" | ".join(headings))
    for match_id, metadata in rows:
        path = STATSBOMB_DATA_DIR / "three-sixty" / f"{match_id}.json"
        try:
            records = load_json_records(path, "three-sixty")
            record_count: str | int = len(records)
            usable: str | int = sum(
                isinstance(record.get("freeze_frame"), list)
                and bool(record["freeze_frame"])
                for record in records
            )
        except VisualizationError as error:
            print(f"Warning: {error}", file=sys.stderr)
            record_count = "invalid JSON"
            usable = "invalid JSON"
        values = (
            str(match_id),
            metadata.date if metadata else "—",
            metadata.competition if metadata else "—",
            metadata.season if metadata else "—",
            metadata.home_team if metadata else "—",
            metadata.away_team if metadata else "—",
            metadata.score if metadata else "—",
            str(record_count),
            str(usable),
        )
        print(" | ".join(values))
    print(f"Available matches shown: {len(rows)}")


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Animate actual event-based StatsBomb 360 freeze frames without "
            "resampling or interpolation."
        )
    )
    parser.add_argument("--match-id", type=int)
    parser.add_argument("--home-team")
    parser.add_argument("--away-team")
    parser.add_argument("--match-date")
    parser.add_argument("--list-available-matches", action="store_true")
    parser.add_argument("--team-filter")
    parser.add_argument("--competition-filter")
    parser.add_argument("--season-filter")
    parser.add_argument("--period", type=int, default=DEFAULT_PERIOD)
    parser.add_argument("--all-events", action="store_true")
    parser.add_argument("--all-period-events", action="store_true")
    parser.add_argument("--maximum-render-frames", type=int, default=0)
    parser.add_argument("--start-minute", type=int)
    parser.add_argument("--end-minute", type=int)
    parser.add_argument(
        "--start-second",
        type=float,
        help="Absolute start second within the period; overrides --start-minute.",
    )
    parser.add_argument(
        "--end-second",
        type=float,
        help="Absolute end second within the period; overrides --end-minute.",
    )
    parser.add_argument(
        "--start-time", help="Period-relative start as MM:SS or HH:MM:SS."
    )
    parser.add_argument(
        "--end-time", help="Period-relative end as MM:SS or HH:MM:SS."
    )
    parser.add_argument(
        "--frame-duration-ms", type=int, default=DEFAULT_FRAME_DURATION_MS
    )
    parser.add_argument(
        "--timing-mode", choices=("fixed", "real-time"), default="real-time"
    )
    parser.add_argument(
        "--show-timing-debug",
        action="store_true",
        help="Append timing mode and encoded GIF hold to frame headers.",
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--minimum-frame-duration-ms", type=int, default=80)
    parser.add_argument("--maximum-frame-duration-ms", type=int, default=1500)
    parser.add_argument(
        "--view",
        choices=("analysis", "camera-overlay", "full-pitch", "visible-area"),
        default="analysis",
        help=(
            "Show the static pitch without visible_area (analysis) or with its "
            "moving polygon overlay (camera-overlay); V2 names remain aliases."
        ),
    )
    parser.add_argument("--theme", choices=("light", "dark"), default="dark")
    parser.add_argument(
        "--team-one-color",
        help="Override the selected theme's default color for team one.",
    )
    parser.add_argument(
        "--team-two-color",
        help="Override the selected theme's default color for team two.",
    )
    parser.add_argument(
        "--team-one-label", help="Override team one's legend-only display label."
    )
    parser.add_argument(
        "--team-two-label", help="Override team two's legend-only display label."
    )
    parser.add_argument(
        "--list-goalkeeper-frames",
        action="store_true",
        help="Print selected frames containing raw keeper=true points.",
    )
    parser.add_argument(
        "--highlight-goalkeepers",
        action="store_true",
        help="Add a contrasting ring around rendered goalkeeper diamonds.",
    )
    parser.add_argument(
        "--apply-orientation-correction",
        action="store_true",
        help=(
            "Correct StatsBomb 360 snapshots recorded 180 degrees rotated "
            "relative to their own event location (see "
            "src/orientation_correction.py for the detection method and "
            "forensic evidence). Off by default so existing and corrected "
            "output can be compared side by side."
        ),
    )
    parser.add_argument(
        "--apply-team-normalization",
        action="store_true",
        help=(
            "Place both teams' events onto one shared pitch coordinate "
            "system: raw StatsBomb coordinates encode each team's own "
            "attack direction as increasing x, so two teams' events are "
            "never on the same scale without this (see "
            "src/team_normalization.py for the forensic evidence). "
            "Independent of --apply-orientation-correction. Off by default "
            "so existing and normalized output can be compared side by "
            "side."
        ),
    )
    parser.add_argument(
        "--apply-team-identity-resolution",
        action="store_true",
        help=(
            "Resolve each non-actor freeze_frame player's ABSOLUTE team "
            "(fixed color for the whole match) instead of the raw "
            "teammate-relative-to-this-event reading, which is unreliable "
            "when StatsBomb reuses one 360 snapshot across events owned by "
            "different teams (see src/team_identity.py for the forensic "
            "evidence). Never changes coordinates, never touches the "
            "actor (always colored by its own event.team). Points whose "
            "absolute team cannot be determined are drawn in a distinct "
            "neutral color rather than guessed. Independent of "
            "--apply-orientation-correction and --apply-team-normalization. "
            "Off by default so existing and resolved output can be "
            "compared side by side."
        ),
    )
    parser.add_argument(
        "--physical-reference-team",
        type=str,
        default=None,
        help=(
            "Override which team --apply-team-normalization treats as the "
            "unmirrored reference, using an exact (case-insensitive) team "
            "name (e.g. 'Spain'). By default the reference team is picked "
            "automatically (whichever team's first 360-matched event is "
            "chronologically earliest, in practice just whichever team "
            "kicks off) -- this default has NO relationship to real "
            "stadium/camera orientation and is only validated for "
            "analytical (e.g. ML feature) use, not for matching a real "
            "broadcast camera. Use this flag when you have independent "
            "evidence (e.g. a broadcast screenshot) of which team actually "
            "appears to attack toward increasing x; see "
            "docs/physical_pitch_orientation_validation.md for the "
            "forensic evidence and an example. Only affects "
            "--apply-team-normalization; has no effect otherwise, and "
            "does not change src/build_statsbomb_360_dataset.py's "
            "unrelated analytical pipeline."
        ),
    )
    parser.add_argument(
        "--presentation-reference-team",
        type=str,
        default=None,
        help=(
            "Deterministic, user-chosen DISPLAY PREFERENCE for which team "
            "is the period-1 baseline -- mechanically identical to "
            "--physical-reference-team (same exact, case-insensitive team "
            "name matching, same period-aware halftime/extra-time swap "
            "applied on top), but explicitly NOT a claim of physical/"
            "broadcast verification. Added 2026-10-22 after finding that "
            "the automatic baseline is an unverifiable coin flip per "
            "match (see "
            "docs/russia_belgium_orientation_regression_2026-10-22.md): "
            "if a match's automatic default happens to look backwards to "
            "you and you have no independent evidence to justify "
            "--physical-reference-team, use this flag instead to get a "
            "stable, reproducible override without asserting it is "
            "physically correct. Reported in the run's own output as "
            "'presentation preference (unverified)', never as calibrated. "
            "Ignored if --physical-reference-team is also given (physical "
            "evidence always takes priority over a bare preference)."
        ),
    )
    parser.add_argument(
        "--display-orientation",
        choices=("camera", "legacy"),
        default="camera",
        help=(
            "Display-only orientation layer used with --apply-team-normalization. "
            "'camera' (default): spectator-facing view with the broadcast camera's "
            "touchline at the bottom, inferred per period from StatsBomb visible_area "
            "geometry (src/broadcast_orientation.py) -- no reference team is chosen; "
            "a verified registry entry or an explicit --physical-reference-team / "
            "--presentation-reference-team still takes priority as the period-1 anchor. "
            "'legacy': the previous behavior (chronologically-first-team baseline plus "
            "a fixed halftime/extra-time rule), kept for comparison. See "
            "docs/final_statsbomb_360_orientation_research.md."
        ),
    )
    parser.add_argument("--event-frame-manifest", type=Path)
    parser.add_argument("--timing-csv", type=Path)
    parser.add_argument("--timing-plot", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    """Discover, select, report, and render actual StatsBomb 360 events."""
    args = build_argument_parser().parse_args()
    metadata_index = load_match_metadata_index()

    if args.list_available_matches:
        try:
            rows = filter_available_matches(
                metadata_index,
                args.team_filter,
                args.competition_filter,
                args.season_filter,
            )
            print_available_matches(rows)
        except VisualizationError as error:
            print(f"Error: {error}", file=sys.stderr)
            return 1
        return 0

    view = normalize_view(args.view)
    theme = get_visual_theme(args.theme)
    if args.view != view:
        print(
            f"Notice: --view {args.view} is deprecated; use --view {view}.",
            file=sys.stderr,
        )
    try:
        match_id = resolve_match_id(
            args.match_id,
            metadata_index,
            args.home_team,
            args.away_team,
            args.match_date,
        )
        match = load_and_join_match(match_id)
        # Step 1 (Problem A): is each event's own freeze_frame internally
        # consistent with its own event.location? Unchanged from before.
        orientation_results = compute_orientation_results(match)
        orientation_summary = summarize_orientation_results(orientation_results)
        # Problem C (rendering-only; a third, independent question -- "which
        # ABSOLUTE team does this point belong to?" -- answered on the RAW,
        # pre-Problem-A match, since Problem A's own correction can make two
        # originally shared/identical snapshots diverge). See
        # src/team_identity.py. Never touches coordinates.
        shared_group_by_event_id = compute_shared_group_info(match)
        raw_events_by_id = {joined.event["id"]: joined.event for joined in match.usable_frames}
        team_identity_summary = summarize_team_identity(match, shared_group_by_event_id)
        # Problem D (display only): camera-side evidence, read from the RAW
        # match with Problem A's per-frame results, before any correction is
        # applied to `match`. See src/broadcast_orientation.py.
        broadcast_inference = infer_broadcast_orientation_for_match(match, orientation_results)
        display_plan: DisplayOrientationPlan | None = None
        display_messages: list[str] = []
        plan_calibration_source: str | None = None
        if args.display_orientation == "camera":
            # Resolved once, on the RAW match (only event team/period fields
            # are read), so the UNCERTAIN-snapshot resolution below can use
            # the same period plan that is later applied for display.
            display_plan, plan_calibration_source, display_messages = resolve_display_plan(
                match_id,
                match,
                broadcast_inference,
                args.physical_reference_team,
                args.presentation_reference_team,
            )
        uncertain_display_counts: dict[str, int] | None = None
        correction_results = orientation_results
        if display_plan is not None:
            correction_results, uncertain_display_counts = display_orientation_results(
                match, orientation_results, broadcast_inference, display_plan,
                frame_reviews=frame_review_corrections(),
            )
        # Team identity (shared with the dataset builder): resolved on the RAW
        # match with the same snapshot orientations the builder uses.
        frame_identities: dict[str, FrameIdentity] | None = None
        if args.apply_team_identity_resolution:
            identity_names = {j.event["team"]["id"]: j.event["team"]["name"] for j in match.usable_frames}
            frame_identities = resolve_match_team_identity(
                [(joined.event, joined.frame_360) for joined in match.usable_frames],
                [result.orientation for result in correction_results],
                orientation_results,
                tuple(identity_names[team_id] for team_id in match_team_ids(match)),
                load_json_records(match.event_path, "event"),
            )
        if args.apply_orientation_correction:
            match = apply_orientation_correction_to_match(match, correction_results)
        else:
            uncertain_display_counts = None
        # Step 2 (Problem B, independent question, applied after Step 1):
        # which of the two teams' coordinate conventions do we standardize
        # on? See src/team_normalization.py. The automatic pick has no
        # relationship to real camera orientation (see
        # docs/physical_pitch_orientation_validation.md); --physical-
        # reference-team lets a caller override it with externally-verified
        # evidence without changing the default (analytical) behavior.
        #
        # A single reference team for the WHOLE match is only physically
        # correct for periods sharing period 1's end arrangement -- real
        # teams swap ends at halftime (and before the second half of extra
        # time), so --physical-reference-team resolves the reference
        # SEPARATELY PER PERIOD (resolve_physical_reference_team_id), never
        # a single fixed id, to avoid rendering later periods backwards.
        # See docs/full_physical_pitch_orientation_audit.md.
        all_usable_events = [joined.event for joined in match.usable_frames]
        reference_by_period: dict[int, int] | None = None
        # Resolution priority: explicit --physical-reference-team (highest,
        # claims verified evidence) > explicit --presentation-reference-team
        # (a deterministic, reproducible DISPLAY PREFERENCE with no
        # verification claim -- added 2026-10-22, see
        # docs/russia_belgium_orientation_regression_2026-10-22.md) > a
        # "verified" entry in configs/physical_orientation_calibration.json
        # (see src/physical_orientation_calibration.py) > the automatic,
        # non-physical default. A registry entry is NEVER applied unless its
        # verification_status is exactly "verified" -- partially_verified/
        # unverified/ambiguous entries are surfaced by review tooling, never
        # silently auto-applied here. See
        # docs/all_matches_physical_orientation_feasibility.md.
        effective_physical_reference_team = args.physical_reference_team
        calibration_source = "cli_override" if effective_physical_reference_team is not None else None
        if effective_physical_reference_team is None and args.presentation_reference_team is not None:
            effective_physical_reference_team = args.presentation_reference_team
            calibration_source = "presentation_preference"
        if effective_physical_reference_team is None:
            calibration_registry = load_calibration_registry()
            registry_team = resolve_usable_physical_reference_team_name(match_id, calibration_registry)
            if registry_team is not None:
                effective_physical_reference_team = registry_team
                calibration_source = "registry"
        if args.display_orientation == "camera":
            periods_present = sorted(
                {integer_event_field(event, "period") for event in all_usable_events}
            )
            calibration_source = plan_calibration_source
            for message in display_messages:
                print(message, file=sys.stderr)
            anchor_id = None
            reference_by_period = display_plan.reference_by_period()
            reference_team_id = reference_by_period.get(periods_present[0], anchor_id)
            team_normalization_summary = summarize_display_plan(match, display_plan)
            if args.apply_team_normalization:
                match = apply_display_plan_to_match(match, display_plan)
        elif effective_physical_reference_team is not None:
            periods_present = sorted(
                {integer_event_field(event, "period") for event in all_usable_events}
            )
            reference_by_period = {
                period: resolve_physical_reference_team_id(
                    all_usable_events, effective_physical_reference_team, period
                )
                for period in periods_present
            }
            reference_team_id = reference_by_period[periods_present[0]]
            team_normalization_summary = summarize_physical_team_normalization(
                match, reference_by_period
            )
            if args.apply_team_normalization:
                match, reference_by_period = apply_team_normalization_to_match_with_physical_calibration(
                    match, all_usable_events, effective_physical_reference_team
                )
        else:
            # No explicit override and no verified registry entry: still
            # apply the period-aware halftime/extra-time-half end swap
            # (root-cause fix, 2026-10-22 -- see
            # docs/full_visualization_orientation_and_clock_audit_2026-10-22.md).
            # This is the renderer's "spectator-facing" presentation
            # convention: a real, universal football-rules fact (teams
            # swap ends) applied consistently to whichever arbitrary
            # automatic baseline was chosen for period 1 -- NOT a claim of
            # physical verification. calibration_source stays None so
            # downstream reporting never describes this as calibrated.
            calibration_source = "spectator_facing_default"
            periods_present = sorted(
                {integer_event_field(event, "period") for event in all_usable_events}
            )
            automatic_baseline = resolve_reference_team_id(all_usable_events)
            reference_by_period = {
                period: apply_period_end_swap(all_usable_events, automatic_baseline, period)
                for period in periods_present
            }
            reference_team_id = reference_by_period[periods_present[0]]
            team_normalization_summary = summarize_physical_team_normalization(
                match, reference_by_period
            )
            if args.apply_team_normalization:
                match, reference_by_period = apply_team_normalization_to_match_spectator_facing(
                    match, all_usable_events
                )
        if (args.all_events or args.all_period_events) and (
            args.start_minute is not None or args.end_minute is not None
        ):
            raise VisualizationError(
                "All-event selection cannot be combined with explicit minute arguments."
            )
        if (args.start_minute is None) != (args.end_minute is None):
            raise VisualizationError(
                "--start-minute and --end-minute must be provided together."
            )
        start_minute = (
            DEFAULT_START_MINUTE if args.start_minute is None else args.start_minute
        )
        end_minute = DEFAULT_END_MINUTE if args.end_minute is None else args.end_minute
        selection = select_actual_event_frames(
            match.usable_frames,
            args.period,
            start_minute,
            end_minute,
            args.start_second,
            args.end_second,
            args.start_time,
            args.end_time,
            args.all_events,
            args.all_period_events,
            args.maximum_render_frames,
        )
        selected = selection.frames
        structural_report = validate_selected_frames(selected)
        team_style = determine_team_style(
            match.usable_frames,
            args.team_one_color or theme.default_team_one,
            args.team_two_color or theme.default_team_two,
            args.team_one_label,
            args.team_two_label,
        )
        direction_period = selection.period or args.period
        direction_inference = infer_attacking_directions(
            match.usable_frames, direction_period, team_style
        )
        if display_plan is not None and args.apply_team_normalization and direction_period in display_plan.transforms:
            team_names_by_id = {
                joined.event["team"]["id"]: joined.event["team"]["name"] for joined in match.usable_frames
            }
            direction_inference = directions_from_display_plan(
                display_plan, direction_period, team_names_by_id, direction_inference
            )
        timing_plan = build_timing_plan(
            selected,
            args.timing_mode,
            args.frame_duration_ms,
            args.playback_speed,
            args.minimum_frame_duration_ms,
            args.maximum_frame_duration_ms,
        )
        density = calculate_temporal_density(
            timing_plan, selection_duration_seconds(selection)
        )
        if args.list_goalkeeper_frames:
            print_goalkeeper_frames(selected)
    except VisualizationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    metadata = metadata_index.get(match_id)
    print_match_summary(match_id, metadata)
    print("Frame source: actual StatsBomb 360 event freeze frames")
    if selection.period is not None:
        print(f"Period: {selection.period}")
    print(f"Requested interval: {selection.interval_label}")
    if selection.interval_start is not None and selection.interval_end is not None:
        print(
            "Resolved interval seconds: "
            f"[{selection.interval_start:g}, {selection.interval_end:g})"
        )
    print(f"Selected real event frames: {len(selected)}")
    print(f"Unique event UUIDs rendered: {len({frame.event['id'] for frame in selected})}")
    print(f"GIF logical frames: {len(selected)}")
    complexity = selection_complexity(len(selected))
    print(f"Estimated output complexity: {complexity}")
    if complexity == "large":
        print("Warning: this large selection may take time and produce a large GIF.")
    print(f"First selected event timestamp: {selected[0].event.get('timestamp')}")
    print(f"Last selected event timestamp: {selected[-1].event.get('timestamp')}")

    output_path = args.output or default_output_path(
        match_id,
        selection,
        theme.name,
        args.timing_mode,
        args.apply_orientation_correction,
        args.apply_team_normalization,
        args.apply_team_identity_resolution,
    )
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    def resolve_optional_path(path: Path | None) -> Path | None:
        if path is None or path.is_absolute():
            return path
        return PROJECT_ROOT / path

    timing_csv_path = resolve_optional_path(args.timing_csv)
    timing_plot_path = resolve_optional_path(args.timing_plot)
    manifest_path = resolve_optional_path(args.event_frame_manifest)
    render_shared_group_by_event_id = (
        shared_group_by_event_id if args.apply_team_identity_resolution else None
    )
    render_raw_events_by_id = raw_events_by_id if args.apply_team_identity_resolution else None
    try:
        if timing_csv_path is not None:
            export_timing_csv(
                timing_csv_path,
                match_id,
                selection.period or 0,
                selected,
                timing_plan,
                args.timing_mode,
            )
        if manifest_path is not None:
            export_event_frame_manifest(manifest_path, match_id, selected, timing_plan)
        if timing_plot_path is not None:
            plot_start = selection.interval_start
            plot_end = selection.interval_end
            if plot_start is None or plot_end is None:
                plot_start = min(timing.match_second for timing in timing_plan)
                plot_end = max(timing.match_second for timing in timing_plan)
                if plot_end <= plot_start:
                    plot_end = plot_start + 1
            save_timing_plot(timing_plot_path, timing_plan, plot_start, plot_end)
        test_frame_path = save_static_test_frame(
            match_id,
            selected,
            output_path,
            view,
            team_style,
            direction_inference,
            args.timing_mode,
            timing_plan,
            theme,
            args.show_timing_debug,
            args.highlight_goalkeepers,
            render_shared_group_by_event_id,
            render_raw_events_by_id,
            frame_identities=frame_identities,
        )
        summary = summarize_segment(selected)
        animation_path = save_animation(
            match_id,
            selected,
            args.frame_duration_ms,
            output_path,
            view,
            team_style,
            direction_inference,
            args.timing_mode,
            timing_plan,
            theme,
            args.show_timing_debug,
            args.highlight_goalkeepers,
            render_shared_group_by_event_id,
            render_raw_events_by_id,
            frame_identities=frame_identities,
        )
    except VisualizationError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if manifest_path is not None:
        with manifest_path.open("r", encoding="utf-8", newline="") as file:
            manifest_rows = sum(1 for _ in csv.DictReader(file))
        if manifest_rows != len(selected):
            print("Error: event-frame manifest row count mismatch.", file=sys.stderr)
            return 1

    print(f"Event records: {match.event_count}")
    print(f"Three-sixty records: {match.three_sixty_count}")
    print(f"Successfully matched records: {match.matched_count}")
    print(f"Usable freeze frames: {match.usable_count}")
    print(
        "Orientation classification (all usable freeze frames; see "
        "src/orientation_correction.py): "
        f"NORMAL={orientation_summary[ORIENTATION_NORMAL]} "
        f"ROTATED_180={orientation_summary[ORIENTATION_ROTATED_180]} "
        f"UNCERTAIN={orientation_summary[ORIENTATION_UNCERTAIN]}"
    )
    print(
        "Orientation correction applied to rendering: "
        f"{'yes' if args.apply_orientation_correction else 'no (use --apply-orientation-correction)'}"
    )
    if uncertain_display_counts is not None:
        print(
            "UNCERTAIN snapshots resolved for display only: "
            f"same-moment duplicate={uncertain_display_counts['resolved_by_duplicate']} "
            f"kick-off own-half={uncertain_display_counts['resolved_by_kickoff_rule']} "
            f"camera+continuity={uncertain_display_counts['resolved_by_camera_continuity']} "
            f"user frame review={uncertain_display_counts['resolved_by_user_frame_review']} "
            f"camera+lean rotated={uncertain_display_counts['resolved_rotated']} "
            f"camera+lean normal={uncertain_display_counts['resolved_normal']} "
            f"left uncorrected={uncertain_display_counts['still_uncertain']} "
            f"(confident copies re-aligned by duplicate consensus={uncertain_display_counts['duplicate_consensus_overrides']}, "
            f"duplicate vs camera disagreements={uncertain_display_counts['duplicate_camera_disagreements']}, "
            f"duplicate conflicts={uncertain_display_counts['duplicate_conflicts']})"
        )
    display_note = ""
    if display_plan is not None:
        display_note = (
            f" Display orientation layer: provenance={display_plan.provenance} "
            f"(camera-side inference status={broadcast_inference.status}: "
            f"{broadcast_inference.explanation}) plan: {display_plan.explanation}."
        )
    if calibration_source == "camera_side_inference":
        reference_source = (
            "camera-side inference from visible_area (spectator-facing, camera touchline at "
            "the bottom; NOT broadcast-verified -- see src/broadcast_orientation.py)"
        )
    elif calibration_source == "cli_override":
        provenance = f"explicit --physical-reference-team={args.physical_reference_team!r}"
        reference_source = (
            f"period-aware PHYSICAL calibration, {provenance} "
            f"(per-period reference_team_id: {reference_by_period}; see "
            "docs/full_physical_pitch_orientation_audit.md for why a single "
            "match-wide reference is wrong for periods after an end swap)"
        )
    elif calibration_source == "registry":
        provenance = (
            f"auto-applied from configs/physical_orientation_calibration.json "
            f"(verified entry, team={effective_physical_reference_team!r}; see "
            "docs/all_matches_physical_orientation_feasibility.md)"
        )
        reference_source = (
            f"period-aware PHYSICAL calibration, {provenance} "
            f"(per-period reference_team_id: {reference_by_period}; see "
            "docs/full_physical_pitch_orientation_audit.md for why a single "
            "match-wide reference is wrong for periods after an end swap)"
        )
    elif calibration_source == "presentation_preference":
        # Deterministic, user-chosen DISPLAY PREFERENCE (2026-10-22) --
        # mechanically period-aware, same as the PHYSICAL branches above,
        # but explicitly and permanently reported as unverified: this flag
        # never claims broadcast/physical evidence, only a reproducible
        # override. See docs/russia_belgium_orientation_regression_2026-10-22.md.
        reference_source = (
            f"presentation PREFERENCE (NOT physically verified), explicit "
            f"--presentation-reference-team={args.presentation_reference_team!r} "
            f"(per-period reference_team_id: {reference_by_period})"
        )
    else:
        # calibration_source == "spectator_facing_default": period-aware,
        # but the period-1 baseline is still the automatic, UNVERIFIED pick
        # -- never describe this as physically verified. See
        # docs/full_visualization_orientation_and_clock_audit_2026-10-22.md.
        reference_source = (
            "spectator-facing DEFAULT convention (period-aware halftime/extra-time "
            "end swap applied to the automatic, chronologically-first-team baseline; "
            "NOT physically verified -- see "
            "docs/full_visualization_orientation_and_clock_audit_2026-10-22.md). "
            f"(per-period reference_team_id: {reference_by_period}). No verified "
            "entry for this match in configs/physical_orientation_calibration.json "
            "-- pass --physical-reference-team if you have evidence, or "
            "--presentation-reference-team for a deterministic preference, or see "
            "docs/all_matches_physical_orientation_feasibility.md to calibrate it."
        )
    print(
        "Team normalization (all usable freeze frames; see "
        "src/team_normalization.py): "
        f"reference_team_id={reference_team_id} ({reference_source}){display_note} "
        f"REFERENCE={team_normalization_summary[TEAM_NORMALIZATION_REFERENCE]} "
        f"MIRRORED={team_normalization_summary[TEAM_NORMALIZATION_MIRRORED]}"
    )
    print(
        "Team normalization applied to rendering: "
        f"{'yes' if args.apply_team_normalization else 'no (use --apply-team-normalization)'}"
    )
    print(
        "Legacy Problem C, byte-identical copies only (all usable freeze frames): "
        f"total={team_identity_summary['total_frames']} "
        f"shared={team_identity_summary['shared_frames']} "
        f"recoverable={team_identity_summary['recoverable_frames']} "
        f"ambiguous={team_identity_summary['ambiguous_frames']}"
    )
    selection_point_counts = summarize_team_identity_points(
        selected, shared_group_by_event_id, raw_events_by_id, team_style
    )
    print(
        "Legacy Problem C point counts for the selected frames (not what is drawn): "
        f"{team_style.team_one}={selection_point_counts[team_style.team_one]} "
        f"{team_style.team_two}={selection_point_counts[team_style.team_two]} "
        f"AMBIGUOUS={selection_point_counts[TEAM_IDENTITY_AMBIGUOUS]}"
    )
    if frame_identities is not None:
        selected_statuses = Counter(
            status for joined in selected for status in frame_identities[joined.event["id"]].statuses
        )
        print(
            "Evidence-based team identity (shared with the dataset builder) for the selected frames: "
            + ", ".join(f"{status}={count}" for status, count in sorted(selected_statuses.items()))
        )
    print(
        "Team identity resolution applied to rendering: "
        f"{'yes' if args.apply_team_identity_resolution else 'no (use --apply-team-identity-resolution)'}"
    )
    print(f"Timing mode: {args.timing_mode}")
    print("Temporal density (actual event-based freeze frames):")
    print(f"  Reporting span: {density.requested_seconds:.3f} match seconds")
    print(f"  Real freeze frames: {density.frame_count}")
    print(f"  Real freeze frames per match second: {density.frames_per_second:.3f}")
    if density.average_gap is not None:
        print(f"  Equivalent average seconds between frames: {density.average_gap:.3f}")
        print(f"  Minimum gap: {density.minimum_gap:.3f} s")
        print(f"  25th percentile gap: {density.percentile_25:.3f} s")
        print(f"  Median gap: {density.median_gap:.3f} s")
        print(f"  75th percentile gap: {density.percentile_75:.3f} s")
        print(f"  Maximum gap: {density.maximum_gap:.3f} s")
    for label, count, percentage in density.bins:
        print(f"  Gaps {label}: {count} ({percentage:.1f}%)")
    print(f"View: {view}")
    print(f"Theme: {theme.name}")
    print("Stable team colors:")
    print(
        f"  {team_style.team_one}: {team_style.team_one_color} "
        f"(legend: {team_style.team_one_label})"
    )
    print(
        f"  {team_style.team_two}: {team_style.team_two_color} "
        f"(legend: {team_style.team_two_label})"
    )
    print(f"Attacking-direction inference: {direction_inference.explanation}")
    print("Structural verification:")
    print(
        "  Player coordinates inside pitch: "
        f"{structural_report.player_coordinate_count - structural_report.player_outside_count}/"
        f"{structural_report.player_coordinate_count} "
        f"({structural_report.player_outside_count} outside across "
        f"{structural_report.player_outside_frame_count} frames in raw data)"
    )
    print(
        "  Ball coordinates inside pitch: "
        f"{structural_report.ball_coordinate_count - structural_report.ball_outside_count}/"
        f"{structural_report.ball_coordinate_count}"
    )
    print(f"First included event timestamp: {summary.first_timestamp}")
    print(f"Last included event timestamp: {summary.last_timestamp}")
    if timing_csv_path is not None:
        print(f"Timing CSV: {timing_csv_path}")
    if manifest_path is not None:
        print(f"Event-frame manifest: {manifest_path}")
    if timing_plot_path is not None:
        print(f"Timing plot: {timing_plot_path}")
    print(f"Static test frame: {test_frame_path}")
    print(f"Animation: {animation_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
