"""Central registry of independently-verified physical pitch-orientation
calibrations for StatsBomb 360 matches.

Single source of truth: `configs/physical_orientation_calibration.json`,
read ONLY through this module. See
docs/all_matches_physical_orientation_feasibility.md for the full
feasibility study and docs/full_physical_pitch_orientation_audit.md for
the first verified case (match 3930160, Spain vs Croatia).

Design choice (deliberately different from src/match_exclusions.py's
load_exclusion_manifest, which raises if the file is absent): the
exclusion manifest is a required, checked-in file that must always exist;
this registry is new and optional -- a fresh checkout, or a checkout
before any match has been calibrated, legitimately has no registry file
at all. A MISSING file therefore means "nothing calibrated yet" (returns
an empty mapping), while a PRESENT-but-malformed file still raises
CalibrationRegistryError immediately, matching the project's established
"fail loudly on a broken config, never silently treat it as empty"
convention for any file that does exist.

A match id simply absent from an existing, valid registry is likewise
NOT an error -- it means "not yet physically calibrated." Callers MUST
treat a missing entry exactly like any other unverified match: fall back
to the automatic, analytical reference-team pick
(src/team_normalization.py's documented default), and report clearly
that physical orientation is unverified. This module never invents,
infers, or guesses a calibration value.

Only entries with verification_status == "verified" are ever eligible for
automatic application (see resolve_usable_physical_reference_team_name).
"partially_verified", "unverified", and "ambiguous" entries are preserved
and readable (e.g. for a human-review tool to display), but are never
silently auto-applied.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "configs" / "physical_orientation_calibration.json"

_REQUIRED_FIELDS = (
    "match_id",
    "period_1_physical_reference_team",
    "verification_status",
    "confidence",
    "evidence_landmark",
    "verification_method",
    "date_verified",
    "report",
)

VERIFICATION_STATUSES = {"verified", "partially_verified", "unverified", "ambiguous"}


class CalibrationRegistryError(Exception):
    """The calibration registry file exists but is malformed or internally
    invalid. Never caught and silenced by this module -- a broken registry
    must stop processing with a clear message, exactly like
    src/match_exclusions.py's ExclusionManifestError.
    """


@dataclass(frozen=True)
class PhysicalCalibration:
    """One documented physical-orientation calibration entry."""

    match_id: int
    period_1_physical_reference_team: str
    verification_status: str
    confidence: str
    evidence_landmark: str
    verification_method: str
    date_verified: str
    report: str
    horizontal_orientation_verified: bool = False
    vertical_orientation_verified: bool = False
    vertical_orientation_note: str = ""
    source_url: str | None = None
    source_description: str = ""
    reviewer_notes: str = ""
    verified_by: str = ""
    period_handling_rule: str = ""


def load_calibration_registry(path: Path = DEFAULT_REGISTRY_PATH) -> dict[int, PhysicalCalibration]:
    """Load and validate the calibration registry.

    Returns {} if `path` does not exist (nothing calibrated yet -- not an
    error). Raises CalibrationRegistryError if `path` exists but is
    malformed, never returning a partially-valid result.
    """
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as error:
        raise CalibrationRegistryError(f"Calibration registry is not valid JSON: {path}: {error}") from error

    if not isinstance(raw, dict) or not isinstance(raw.get("calibrations"), list):
        raise CalibrationRegistryError(
            f"Calibration registry must be a JSON object with a 'calibrations' list: {path}"
        )

    result: dict[int, PhysicalCalibration] = {}
    for index, entry in enumerate(raw["calibrations"]):
        if not isinstance(entry, dict):
            raise CalibrationRegistryError(f"Calibration registry entry {index} is not an object: {path}")

        missing = [field for field in _REQUIRED_FIELDS if field not in entry]
        if missing:
            raise CalibrationRegistryError(
                f"Calibration registry entry {index} is missing required field(s) {missing}: {path}"
            )

        match_id = entry["match_id"]
        if isinstance(match_id, bool) or not isinstance(match_id, int):
            raise CalibrationRegistryError(
                f"Calibration registry entry {index} has a non-integer match_id: {match_id!r} ({path})"
            )
        if match_id in result:
            raise CalibrationRegistryError(f"Calibration registry has a duplicate match_id {match_id}: {path}")

        status = entry["verification_status"]
        if status not in VERIFICATION_STATUSES:
            raise CalibrationRegistryError(
                f"Calibration registry entry {index} has an invalid verification_status {status!r}; "
                f"must be one of {sorted(VERIFICATION_STATUSES)}: {path}"
            )

        result[match_id] = PhysicalCalibration(
            match_id=match_id,
            period_1_physical_reference_team=str(entry["period_1_physical_reference_team"]),
            verification_status=status,
            confidence=str(entry["confidence"]),
            evidence_landmark=str(entry["evidence_landmark"]),
            verification_method=str(entry["verification_method"]),
            date_verified=str(entry["date_verified"]),
            report=str(entry["report"]),
            horizontal_orientation_verified=bool(entry.get("horizontal_orientation_verified", False)),
            vertical_orientation_verified=bool(entry.get("vertical_orientation_verified", False)),
            vertical_orientation_note=str(entry.get("vertical_orientation_note", "")),
            source_url=entry.get("source_url"),
            source_description=str(entry.get("source_description", "")),
            reviewer_notes=str(entry.get("reviewer_notes", "")),
            verified_by=str(entry.get("verified_by", "")),
            period_handling_rule=str(entry.get("period_handling_rule", "")),
        )
    return result


# --------------------------------------------------------------------------
# Period-level review decisions (2026-10-23).
#
# A reviewer may confirm the DISPLAY orientation of one period (e.g. "the
# proposed render of period 2 is right") without supplying independent
# footage, or without seeing the other periods at all. Such a decision is
# not a physical calibration of the whole match and must never be read as
# one, so it lives in a separate optional top-level list, "period_reviews",
# never in "calibrations". Each entry is scoped to exactly one period.
# --------------------------------------------------------------------------

EVIDENCE_USER_RENDER_REVIEW = "user_review_of_renders"
EVIDENCE_INDEPENDENT_VIDEO = "independent_video"
PERIOD_REVIEW_EVIDENCE_TYPES = {EVIDENCE_USER_RENDER_REVIEW, EVIDENCE_INDEPENDENT_VIDEO}

_REQUIRED_PERIOD_REVIEW_FIELDS = (
    "match_id",
    "period",
    "display_reference_team",
    "evidence_type",
    "reviewed_by",
    "date_reviewed",
)


@dataclass(frozen=True)
class PeriodReview:
    """One reviewer decision about ONE period's display orientation.

    `display_reference_team` is the team that attacks toward increasing
    display x (screen left -> right) in that period, in the camera-at-bottom
    presentation -- i.e. the team left unrotated. An entry with
    evidence_type "independent_video" must name its source_url; it is
    still period-scoped and is still not a match calibration.
    """

    match_id: int
    period: int
    display_reference_team: str
    evidence_type: str
    reviewed_by: str
    date_reviewed: str
    decision: str = ""
    statement: str = ""
    source_url: str | None = None
    notes: str = ""
    report: str = ""

    @property
    def independently_verified(self) -> bool:
        return self.evidence_type == EVIDENCE_INDEPENDENT_VIDEO and bool(self.source_url)


def load_period_reviews(path: Path = DEFAULT_REGISTRY_PATH) -> dict[int, dict[int, PeriodReview]]:
    """Load the optional "period_reviews" list: {match_id: {period: review}}.

    Missing file or missing list -> {}. Malformed entries raise
    CalibrationRegistryError, like the calibrations list.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as error:
        raise CalibrationRegistryError(f"Calibration registry is not valid JSON: {path}: {error}") from error
    entries = raw.get("period_reviews", []) if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise CalibrationRegistryError(f"'period_reviews' must be a list when present: {path}")

    result: dict[int, dict[int, PeriodReview]] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CalibrationRegistryError(f"period_reviews entry {index} is not an object: {path}")
        missing = [field for field in _REQUIRED_PERIOD_REVIEW_FIELDS if field not in entry]
        if missing:
            raise CalibrationRegistryError(f"period_reviews entry {index} is missing field(s) {missing}: {path}")
        match_id, period = entry["match_id"], entry["period"]
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (match_id, period)) or not 1 <= period <= 5:
            raise CalibrationRegistryError(f"period_reviews entry {index} has an invalid match_id/period: {path}")
        if entry["evidence_type"] not in PERIOD_REVIEW_EVIDENCE_TYPES:
            raise CalibrationRegistryError(
                f"period_reviews entry {index} has evidence_type {entry['evidence_type']!r}; "
                f"must be one of {sorted(PERIOD_REVIEW_EVIDENCE_TYPES)}: {path}"
            )
        if entry["evidence_type"] == EVIDENCE_INDEPENDENT_VIDEO and not entry.get("source_url"):
            raise CalibrationRegistryError(
                f"period_reviews entry {index} claims independent_video without a source_url: {path}"
            )
        if period in result.get(match_id, {}):
            raise CalibrationRegistryError(f"period_reviews has a duplicate (match {match_id}, period {period}): {path}")
        result.setdefault(match_id, {})[period] = PeriodReview(
            match_id=match_id,
            period=period,
            display_reference_team=str(entry["display_reference_team"]),
            evidence_type=entry["evidence_type"],
            reviewed_by=str(entry["reviewed_by"]),
            date_reviewed=str(entry["date_reviewed"]),
            decision=str(entry.get("decision", "")),
            statement=str(entry.get("statement", "")),
            source_url=entry.get("source_url"),
            notes=str(entry.get("notes", "")),
            report=str(entry.get("report", "")),
        )
    return result


def resolve_usable_physical_reference_team_name(
    match_id: int, registry: dict[int, PhysicalCalibration]
) -> str | None:
    """Return the period-1 physical reference team name ONLY for an entry
    whose verification_status is exactly "verified".

    "partially_verified", "unverified", and "ambiguous" entries are never
    auto-applied -- they exist in the registry so a human-review tool can
    surface and resolve them, not so the renderer silently trusts them.
    Returns None if the match has no entry at all, or its entry is not
    fully verified; callers must treat None exactly like "not calibrated".
    """
    calibration = registry.get(match_id)
    if calibration is None or calibration.verification_status != "verified":
        return None
    return calibration.period_1_physical_reference_team


# --------------------------------------------------------------------------
# Frame-level review decisions (2026-10-24).
#
# A reviewer may judge ONE snapshot whose orientation the data could not
# decide (Problem A UNCERTAIN, no display rule applicable). The decision is
# stored per snapshot -- all events sharing it -- as the correction relative
# to the snapshot AS STORED in its own event's frame ("NONE" keeps it,
# "ROTATED_180" rotates it). That correction is independent of the period
# display plan, so it stays valid if the plan's absolute frame changes.
# Separate list "frame_reviews"; never a calibration, never video evidence.
# --------------------------------------------------------------------------

FRAME_SNAPSHOT_CORRECTIONS = {"NONE", "ROTATED_180"}
_REQUIRED_FRAME_REVIEW_FIELDS = (
    "match_id", "period", "event_ids", "snapshot_correction", "evidence_type", "reviewed_by", "date_reviewed",
)


@dataclass(frozen=True)
class FrameReview:
    """One reviewer decision about one snapshot (all events that share it)."""

    match_id: int
    period: int
    event_ids: tuple[str, ...]
    snapshot_correction: str
    evidence_type: str
    reviewed_by: str
    date_reviewed: str
    timestamp: str = ""
    selected_panel: str = ""
    sheet: str = ""
    open_issue: str = ""
    notes: str = ""


def load_frame_reviews(path: Path = DEFAULT_REGISTRY_PATH) -> dict[str, FrameReview]:
    """Load the optional "frame_reviews" list as {event_id: review}.

    Missing file or list -> {}. Malformed entries raise CalibrationRegistryError.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as error:
        raise CalibrationRegistryError(f"Calibration registry is not valid JSON: {path}: {error}") from error
    entries = raw.get("frame_reviews", []) if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise CalibrationRegistryError(f"'frame_reviews' must be a list when present: {path}")
    result: dict[str, FrameReview] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CalibrationRegistryError(f"frame_reviews entry {index} is not an object: {path}")
        missing = [field for field in _REQUIRED_FRAME_REVIEW_FIELDS if field not in entry]
        if missing:
            raise CalibrationRegistryError(f"frame_reviews entry {index} is missing field(s) {missing}: {path}")
        event_ids = entry["event_ids"]
        if not isinstance(event_ids, list) or not event_ids or not all(isinstance(e, str) and e for e in event_ids):
            raise CalibrationRegistryError(f"frame_reviews entry {index} needs a non-empty list of event ids: {path}")
        if entry["snapshot_correction"] not in FRAME_SNAPSHOT_CORRECTIONS:
            raise CalibrationRegistryError(
                f"frame_reviews entry {index} has snapshot_correction {entry['snapshot_correction']!r}; "
                f"must be one of {sorted(FRAME_SNAPSHOT_CORRECTIONS)}: {path}"
            )
        if entry["evidence_type"] not in PERIOD_REVIEW_EVIDENCE_TYPES:
            raise CalibrationRegistryError(f"frame_reviews entry {index} has an invalid evidence_type: {path}")
        review = FrameReview(
            match_id=int(entry["match_id"]), period=int(entry["period"]), event_ids=tuple(event_ids),
            snapshot_correction=entry["snapshot_correction"], evidence_type=entry["evidence_type"],
            reviewed_by=str(entry["reviewed_by"]), date_reviewed=str(entry["date_reviewed"]),
            timestamp=str(entry.get("timestamp", "")), selected_panel=str(entry.get("selected_panel", "")),
            sheet=str(entry.get("sheet", "")), open_issue=str(entry.get("open_issue", "")),
            notes=str(entry.get("notes", "")),
        )
        for event_id in event_ids:
            if event_id in result:
                raise CalibrationRegistryError(f"frame_reviews lists event {event_id} twice: {path}")
            result[event_id] = review
    return result
