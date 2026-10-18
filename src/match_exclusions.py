"""Central registry of StatsBomb matches explicitly excluded from processing.

Single source of truth: `configs/excluded_matches.json` is read ONLY through
this module. src/build_statsbomb_360_dataset.py and scripts/audit_dataset.py
both import from here rather than hard-coding match ids, so a match is
excluded consistently everywhere or nowhere -- never inconsistently between
the dataset builder and the audit.

Design choice (deliberate, not an oversight): every failure mode -- a
missing file, invalid JSON, the wrong top-level shape, a malformed entry, a
non-integer match_id, or a duplicate match_id -- raises `ExclusionManifestError`
rather than silently falling back to "no exclusions." A broken manifest is a
configuration bug that must be visible immediately; treating it as "empty"
would silently re-expose already-diagnosed corrupted/unusable matches to
normal processing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "configs" / "excluded_matches.json"

_REQUIRED_FIELDS = ("match_id", "reason", "file", "evidence", "date_investigated", "report")


class ExclusionManifestError(Exception):
    """The exclusion manifest is missing, malformed, or internally invalid.

    Never caught and silenced by this module or its consumers -- a broken
    manifest must stop processing with a clear message, not be treated as
    an empty exclusion list.
    """


class ExcludedMatchError(Exception):
    """Raised when processing is attempted on an explicitly excluded match.

    Distinct from any exception the match's own raw data might otherwise
    raise (e.g. JSONDecodeError), so callers (notably the audit script) can
    tell "deliberately excluded, documented" apart from "unexpectedly
    failed" -- the two must never be conflated in a report.
    """

    def __init__(self, excluded: "ExcludedMatch") -> None:
        self.excluded = excluded
        super().__init__(
            f"Match {excluded.match_id} is explicitly excluded ({excluded.reason}): "
            f"see {excluded.report}"
        )


@dataclass(frozen=True)
class ExcludedMatch:
    """One documented exclusion entry."""

    match_id: int
    reason: str
    file: str
    evidence: str
    date_investigated: str
    report: str


def load_exclusion_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[int, ExcludedMatch]:
    """Load and validate the exclusion manifest. Raises ExclusionManifestError
    on any structural problem; never returns a partially-valid result.
    """
    if not path.exists():
        raise ExclusionManifestError(f"Exclusion manifest not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as error:
        raise ExclusionManifestError(f"Exclusion manifest is not valid JSON: {path}: {error}") from error

    if not isinstance(raw, dict) or not isinstance(raw.get("excluded_matches"), list):
        raise ExclusionManifestError(
            f"Exclusion manifest must be a JSON object with an 'excluded_matches' list: {path}"
        )

    result: dict[int, ExcludedMatch] = {}
    for index, entry in enumerate(raw["excluded_matches"]):
        if not isinstance(entry, dict):
            raise ExclusionManifestError(f"Exclusion manifest entry {index} is not an object: {path}")

        missing = [field for field in _REQUIRED_FIELDS if field not in entry]
        if missing:
            raise ExclusionManifestError(
                f"Exclusion manifest entry {index} is missing required field(s) {missing}: {path}"
            )

        match_id = entry["match_id"]
        if isinstance(match_id, bool) or not isinstance(match_id, int):
            raise ExclusionManifestError(
                f"Exclusion manifest entry {index} has a non-integer match_id: {match_id!r} ({path})"
            )
        if match_id in result:
            raise ExclusionManifestError(f"Exclusion manifest has a duplicate match_id {match_id}: {path}")

        result[match_id] = ExcludedMatch(
            match_id=match_id,
            reason=str(entry["reason"]),
            file=str(entry["file"]),
            evidence=str(entry["evidence"]),
            date_investigated=str(entry["date_investigated"]),
            report=str(entry["report"]),
        )

    return result


def check_not_excluded(match_id: int, manifest: dict[int, ExcludedMatch]) -> None:
    """Raise ExcludedMatchError if `match_id` is in the manifest; no-op otherwise."""
    excluded = manifest.get(match_id)
    if excluded is not None:
        raise ExcludedMatchError(excluded)
