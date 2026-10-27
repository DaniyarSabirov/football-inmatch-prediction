"""Comprehensive, reproducible StatsBomb category-taxonomy and frequency audit.

Reads ONLY raw JSON under data/raw/statsbomb/ and the project's own exclusion
manifest (configs/excluded_matches.json, via src/match_exclusions.py). Never
modifies data/raw, data/interim, data/processed, or any existing prediction
code. Writes only to results/data_taxonomy/ (JSON + Excel workbook).

Scope decision (documented, not silent): the "raw" population audited here
is the 426 matches that have a local StatsBomb 360 (three-sixty) JSON file
-- the same population src/build_statsbomb_360_dataset.py and
scripts/audit_dataset.py already draw from, since the thesis pipeline
requires paired event+360 data. The local checkout also contains 4,235
total event files (most without any 360 data); those extra ~3,809 matches
are out of scope for this audit and are only reported as a count for
context. "Eligible" = raw minus the matches listed in
configs/excluded_matches.json.

Two independent aggregations are kept throughout (raw vs. eligible) so
every statistic can be reported both ways without re-reading any file
twice.

Usage:
    .venv/bin/python scripts/audit_statsbomb_category_taxonomy.py
    .venv/bin/python scripts/audit_statsbomb_category_taxonomy.py --sample 40
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_exclusions import DEFAULT_MANIFEST_PATH, load_exclusion_manifest  # noqa: E402
from src.build_statsbomb_360_dataset import MEANINGFUL_EVENT_TYPES  # noqa: E402

STATSBOMB_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
EVENTS_DIR = STATSBOMB_DATA_DIR / "events"
FRAMES_DIR = STATSBOMB_DATA_DIR / "three-sixty"
MATCHES_DIR = STATSBOMB_DATA_DIR / "matches"
LINEUPS_DIR = STATSBOMB_DATA_DIR / "lineups"

OUTPUT_DIR = PROJECT_ROOT / "results" / "data_taxonomy"
OUTPUT_JSON = OUTPUT_DIR / "statsbomb_complete_category_audit_data.json"
OUTPUT_XLSX = OUTPUT_DIR / "statsbomb_complete_category_audit.xlsx"

HIGH_CARDINALITY_THRESHOLD = 300  # fields with more raw unique values move to the overflow sheet
RESERVOIR_CAP = 50_000  # numeric-field median approximation cap; exact below this size

IDENTIFIER_LEAF_NAMES = {"id", "uuid", "event_uuid", "key"}
FORCE_CATEGORICAL_NUMERIC_LEAVES = {"formation"}

# Event types currently selected as the pipeline's 8 target/current classes.
# Re-imported directly from the production module -- never redefined here --
# so this audit cannot silently drift from the code it is auditing.
EIGHT_CLASSES = set(MEANINGFUL_EVENT_TYPES)

# Domain judgment on whether each discovered primary event type represents
# an on-ball action potentially relevant to a "next meaningful event"
# prediction task, independent of whether it is currently included. Applied
# only to types actually found in the data (see RELEVANCE_FALLBACK).
RELEVANCE_NOTES = {
    "Pass": "On-ball action; in the current 8 classes.",
    "Carry": "On-ball action; in the current 8 classes.",
    "Dribble": "On-ball action; in the current 8 classes.",
    "Shot": "On-ball action; in the current 8 classes. Cannot currently distinguish goal vs. non-goal (see Goals Audit).",
    "Miscontrol": "On-ball action; in the current 8 classes.",
    "Ball Recovery": "On-ball action; in the current 8 classes.",
    "Clearance": "On-ball action; in the current 8 classes.",
    "Dispossessed": "On-ball action; in the current 8 classes.",
    "Ball Receipt*": "On-ball action (receiving a pass); excluded from the 8 classes despite being one of the most frequent event types. Likely the single largest omission -- see Recommendations.",
    "Pressure": "Off-ball/concurrent annotation (applied to the pressuring player while another event is the on-ball action); does not itself represent a ball-state transition. Reasonable to exclude.",
    "Duel": "On-ball/contested action (tackle or aerial); currently excluded. Candidate for inclusion in a broader class set.",
    "Interception": "On-ball action (actively winning the ball); currently excluded. Candidate for inclusion.",
    "Block": "On-ball defensive action; currently excluded. Candidate for inclusion.",
    "50/50": "On-ball contested-ball action; currently excluded. Candidate for inclusion, but rare.",
    "Goal Keeper": "On-ball action for the goalkeeper (save/claim/distribution); currently excluded. Candidate for inclusion, though goalkeeper actions are a specialised sub-domain.",
    "Dribbled Past": "Defender-perspective complement of Dribble; does not itself describe the ball carrier's action. Low relevance as a standalone next-event label.",
    "Error": "Rare player-error annotation; usually co-occurs with another primary event. Low relevance, very rare.",
    "Shield": "On-ball shielding action (no progression); low frequency, moderate relevance.",
    "Foul Committed": "Rule/administrative event, not itself a ball-state transition. Not recommended as a target class.",
    "Foul Won": "Rule/administrative event. Not recommended as a target class.",
    "Offside": "Rule/administrative event. Not recommended as a target class.",
    "Own Goal For": "Represents a goal (credited to this team); paired with Own Goal Against. See Goals Audit for how to interpret this without double-counting.",
    "Own Goal Against": "Represents a goal (conceded by this team); paired with Own Goal For. See Goals Audit.",
    "Substitution": "Administrative/stoppage event, not an on-ball action. Not recommended as a target class.",
    "Player On": "Administrative event. Not recommended as a target class.",
    "Player Off": "Administrative event. Not recommended as a target class.",
    "Tactical Shift": "Administrative/formation-change event. Not recommended as a target class.",
    "Starting XI": "Administrative (lineup) event, one per team per match. Not recommended as a target class.",
    "Half Start": "Administrative/period-boundary event. Not recommended as a target class.",
    "Half End": "Administrative/period-boundary event. Not recommended as a target class.",
    "Bad Behaviour": "Rule/administrative (card) event. Not recommended as a target class.",
    "Injury Stoppage": "Administrative/stoppage event. Not recommended as a target class.",
    "Referee Ball-Drop": "Rare administrative restart event. Not recommended as a target class.",
    "Camera On": "Broadcast/administrative metadata event, not a football action. Not recommended as a target class.",
    "Camera off": "Broadcast/administrative metadata event, not a football action. Not recommended as a target class.",
}
RELEVANCE_FALLBACK = "Not previously catalogued by this audit's domain notes; inspect manually before using as a target class."


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Generic recursive categorical/field discovery
# ---------------------------------------------------------------------------


class FieldAggregator:
    """Accumulates generic per-field statistics over an arbitrary population
    of JSON "records" (events, or 360 freeze-frame records). Built once for
    "raw" and once for "eligible" so every stat can be reported both ways.
    """

    def __init__(self) -> None:
        self.container_counts: Counter = Counter()
        self.field_types: dict[str, str] = {}
        self.field_present: Counter = Counter()
        self.field_values: dict[str, Counter] = defaultdict(Counter)
        self.field_value_matches: dict[str, dict[str, set]] = defaultdict(lambda: defaultdict(set))
        self.field_matches: dict[str, set] = defaultdict(set)
        self.numeric_stats: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "sum": 0.0, "sumsq": 0.0, "min": None, "max": None, "reservoir": []}
        )
        self.enum_id_name_map: dict[str, Counter] = defaultdict(Counter)
        self.list_empty_counts: Counter = Counter()
        self.list_object_element_counts: Counter = Counter()
        self.records_total = 0


def _classify_and_record(path: str, value, agg: FieldAggregator, match_id: int) -> None:
    leaf_name = path.rsplit(".", 1)[-1]
    if isinstance(value, bool):
        ftype = "boolean"
        val_str = str(value)
    elif isinstance(value, (int, float)):
        if leaf_name in FORCE_CATEGORICAL_NUMERIC_LEAVES:
            ftype, val_str = "categorical", str(value)
        else:
            ftype, val_str = "numeric", None
    elif isinstance(value, str):
        if leaf_name == "timestamp":
            ftype, val_str = "timestamp", None
        elif leaf_name in IDENTIFIER_LEAF_NAMES or leaf_name.endswith("_id"):
            ftype, val_str = "identifier", None
        else:
            ftype, val_str = "categorical", value
    else:
        ftype, val_str = "other", repr(value)

    agg.field_types[path] = ftype
    agg.field_present[path] += 1
    agg.field_matches[path].add(match_id)

    if ftype in ("categorical", "boolean"):
        agg.field_values[path][val_str] += 1
        agg.field_value_matches[path][val_str].add(match_id)
    elif ftype == "numeric":
        st = agg.numeric_stats[path]
        v = float(value)
        st["count"] += 1
        st["sum"] += v
        st["sumsq"] += v * v
        st["min"] = v if st["min"] is None else min(st["min"], v)
        st["max"] = v if st["max"] is None else max(st["max"], v)
        reservoir = st["reservoir"]
        if len(reservoir) < RESERVOIR_CAP:
            reservoir.append(v)
        else:
            j = random.randint(0, st["count"] - 1)
            if j < RESERVOIR_CAP:
                reservoir[j] = v


def _record_enum_id(path: str, id_value, name_value, agg: FieldAggregator, match_id: int, container_path: str) -> None:
    agg.field_types[path] = "enum_id"
    agg.field_present[path] += 1
    agg.field_matches[path].add(match_id)
    agg.enum_id_name_map[container_path][(id_value, name_value)] += 1


def _handle_list(path: str, items: list, agg: FieldAggregator, match_id: int) -> None:
    if not items:
        agg.list_empty_counts[path] += 1
        agg.field_types.setdefault(path, "list_empty_or_scalar")
        # An empty list is a PRESENT (non-null) field, just with zero elements --
        # distinct from the key being absent/null. Must still count toward the
        # field's own presence so Field Inventory never reports an empty-but-
        # present list as "missing".
        agg.field_present[path] += 1
        agg.field_matches[path].add(match_id)
        return
    if all(isinstance(e, dict) for e in items):
        container_path = path + "[]"
        agg.field_types[path] = "list_of_objects"
        agg.list_object_element_counts[path] += len(items)
        # The list field itself (e.g. "freeze_frame", "tactics.lineup") is
        # present once per record, independent of how many elements it has --
        # must be counted here, not only via its elements' container_counts,
        # or Field Inventory would show it as 100% "missing".
        agg.field_present[path] += 1
        agg.field_matches[path].add(match_id)
        for element in items:
            agg.container_counts[container_path] += 1
            _walk(element, container_path, agg, match_id)
    else:
        agg.field_types[path] = "list_of_scalars"
        agg.field_present[path] += 1
        agg.field_matches[path].add(match_id)


def _walk(obj: dict, path: str, agg: FieldAggregator, match_id: int) -> None:
    if not obj:
        return
    has_id_name = "id" in obj and "name" in obj
    for key, value in obj.items():
        if value is None:
            continue
        child_path = f"{path}.{key}" if path else key
        if has_id_name and key == "id":
            _record_enum_id(child_path, value, obj.get("name"), agg, match_id, path)
            continue
        if isinstance(value, dict):
            agg.container_counts[child_path] += 1
            _walk(value, child_path, agg, match_id)
        elif isinstance(value, list):
            _handle_list(child_path, value, agg, match_id)
        else:
            _classify_and_record(child_path, value, agg, match_id)


def ingest_record(record: dict, agg: FieldAggregator, match_id: int) -> None:
    agg.container_counts[""] += 1
    agg.records_total += 1
    _walk(record, "", agg, match_id)


# ---------------------------------------------------------------------------
# Match population / metadata
# ---------------------------------------------------------------------------


def load_match_metadata_index() -> dict[int, dict]:
    index: dict[int, dict] = {}
    for season_file in MATCHES_DIR.glob("*/*.json"):
        try:
            with season_file.open(encoding="utf-8") as f:
                records = json.load(f)
        except Exception:  # noqa: BLE001
            continue
        for rec in records:
            mid = rec.get("match_id")
            if not isinstance(mid, int):
                continue
            index[mid] = {
                "match_id": mid,
                "competition": rec.get("competition", {}).get("competition_name"),
                "season": rec.get("season", {}).get("season_name"),
                "home_team_id": rec.get("home_team", {}).get("home_team_id"),
                "home_team_name": rec.get("home_team", {}).get("home_team_name"),
                "away_team_id": rec.get("away_team", {}).get("away_team_id"),
                "away_team_name": rec.get("away_team", {}).get("away_team_name"),
                "home_score": rec.get("home_score"),
                "away_score": rec.get("away_score"),
                "match_date": rec.get("match_date"),
                "competition_stage": rec.get("competition_stage", {}).get("name"),
            }
    return index


def discover_raw_population() -> list[int]:
    return sorted(int(p.stem) for p in FRAMES_DIR.glob("*.json") if p.stem.isdigit())


def count_total_event_files() -> int:
    return sum(1 for _ in EVENTS_DIR.glob("*.json"))


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------


def mean_median_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "std": None, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def run_audit(sample: int | None, seed: int) -> dict:
    # Fixed seed for the numeric-field reservoir sampling in _classify_and_record
    # (module-level `random.randint` calls once a field's count exceeds
    # RESERVOIR_CAP) -- without this, median_approx would differ between two
    # runs over identical data, violating the determinism requirement for
    # this audit. Match processing order is already deterministic (sorted
    # match ids), so fixing this one seed makes the entire run reproducible.
    random.seed(20261020)

    manifest = load_exclusion_manifest(DEFAULT_MANIFEST_PATH)
    excluded_ids = set(manifest.keys())

    raw_ids = discover_raw_population()
    if sample is not None and sample < len(raw_ids):
        rng = random.Random(seed)
        raw_ids = sorted(rng.sample(raw_ids, sample))
    eligible_ids_target = set(raw_ids) - excluded_ids

    total_event_files = count_total_event_files()
    match_meta = load_match_metadata_index()

    agg_events_raw = FieldAggregator()
    agg_events_eligible = FieldAggregator()
    agg_360_raw = FieldAggregator()
    agg_360_eligible = FieldAggregator()

    type_count_by_match: dict[int, Counter] = {}
    eight_class_current_with_360_by_match: dict[int, Counter] = {}

    goal_tally = {
        "shot_goal_raw": 0, "shot_goal_eligible": 0,
        "own_goal_for_raw": 0, "own_goal_for_eligible": 0,
        "own_goal_against_raw": 0, "own_goal_against_eligible": 0,
    }
    shot_goals_by_match_team: dict[int, Counter] = defaultdict(Counter)
    own_goal_credit_by_match_team: dict[int, Counter] = defaultdict(Counter)

    match_rows: list[dict] = []
    malformed_files: list[dict] = []
    matches_loaded_ok = 0

    snapshot_counts_per_match: dict[int, int] = {}
    player_entry_counts_per_match: dict[int, int] = {}

    for mid in raw_ids:
        is_excluded = mid in excluded_ids
        row = {
            "match_id": mid,
            **{k: v for k, v in (match_meta.get(mid) or {}).items() if k != "match_id"},
            "eligible": False,
            "exclusion_reason": manifest[mid].reason if is_excluded else None,
            "load_status": None,
            "total_events": None,
            "total_three_sixty_records": None,
            "total_freeze_frame_players": None,
        }

        events_path = EVENTS_DIR / f"{mid}.json"
        frames_path = FRAMES_DIR / f"{mid}.json"

        events, frames = None, None
        for path in (events_path, frames_path):
            try:
                with path.open(encoding="utf-8") as f:
                    data = json.load(f)
                if path is events_path:
                    events = data
                else:
                    frames = data
            except Exception as error:  # noqa: BLE001
                malformed_files.append(
                    {"match_id": mid, "file": str(path), "error_type": type(error).__name__, "error_message": str(error)}
                )
                events = frames = None
                break

        if events is None or frames is None:
            row["load_status"] = "failed_to_load" if not is_excluded else "excluded_not_loaded"
            match_rows.append(row)
            continue

        matches_loaded_ok += 1
        is_eligible = (not is_excluded) and mid in eligible_ids_target
        row["eligible"] = is_eligible
        row["load_status"] = "excluded" if is_excluded else "ok"
        row["total_events"] = len(events)
        row["total_three_sixty_records"] = len(frames)

        frame_ids = {fr.get("event_uuid") for fr in frames}

        type_counter = Counter()
        eight_counter = Counter()
        for event in events:
            etype = event.get("type", {}).get("name")
            type_counter[etype] += 1

            ingest_record(event, agg_events_raw, mid)
            if is_eligible:
                ingest_record(event, agg_events_eligible, mid)

            if etype in EIGHT_CLASSES and event.get("id") in frame_ids:
                eight_counter[etype] += 1

            if etype == "Shot" and (event.get("shot") or {}).get("outcome", {}).get("name") == "Goal":
                goal_tally["shot_goal_raw"] += 1
                shot_goals_by_match_team[mid][event.get("team", {}).get("id")] += 1
                if is_eligible:
                    goal_tally["shot_goal_eligible"] += 1
            elif etype == "Own Goal For":
                goal_tally["own_goal_for_raw"] += 1
                own_goal_credit_by_match_team[mid][event.get("team", {}).get("id")] += 1
                if is_eligible:
                    goal_tally["own_goal_for_eligible"] += 1
            elif etype == "Own Goal Against":
                goal_tally["own_goal_against_raw"] += 1
                if is_eligible:
                    goal_tally["own_goal_against_eligible"] += 1

        type_count_by_match[mid] = type_counter
        eight_class_current_with_360_by_match[mid] = eight_counter

        total_players = 0
        for fr in frames:
            ingest_record(fr, agg_360_raw, mid)
            if is_eligible:
                ingest_record(fr, agg_360_eligible, mid)
            total_players += len(fr.get("freeze_frame") or [])

        row["total_freeze_frame_players"] = total_players
        snapshot_counts_per_match[mid] = len(frames)
        player_entry_counts_per_match[mid] = total_players

        match_rows.append(row)

    log(
        f"Processed {matches_loaded_ok}/{len(raw_ids)} raw matches "
        f"({len(eligible_ids_target & set(raw_ids))} eligible), {len(malformed_files)} malformed."
    )

    return {
        "manifest": manifest,
        "excluded_ids": excluded_ids,
        "raw_ids": raw_ids,
        "eligible_ids": sorted(eligible_ids_target & set(raw_ids)),
        "total_event_files": total_event_files,
        "match_meta": match_meta,
        "agg_events_raw": agg_events_raw,
        "agg_events_eligible": agg_events_eligible,
        "agg_360_raw": agg_360_raw,
        "agg_360_eligible": agg_360_eligible,
        "type_count_by_match": type_count_by_match,
        "eight_class_current_with_360_by_match": eight_class_current_with_360_by_match,
        "goal_tally": goal_tally,
        "shot_goals_by_match_team": shot_goals_by_match_team,
        "own_goal_credit_by_match_team": own_goal_credit_by_match_team,
        "match_rows": match_rows,
        "malformed_files": malformed_files,
        "snapshot_counts_per_match": snapshot_counts_per_match,
        "player_entry_counts_per_match": player_entry_counts_per_match,
    }


# ---------------------------------------------------------------------------
# Derived statistics
# ---------------------------------------------------------------------------


def build_event_type_table(audit: dict) -> list[dict]:
    raw_ids = audit["raw_ids"]
    eligible_ids = set(audit["eligible_ids"])
    type_count_by_match = audit["type_count_by_match"]

    all_types = set()
    for c in type_count_by_match.values():
        all_types.update(c.keys())

    raw_total_events = sum(sum(c.values()) for mid, c in type_count_by_match.items())
    eligible_total_events = sum(sum(c.values()) for mid, c in type_count_by_match.items() if mid in eligible_ids)

    rows = []
    for t in sorted(all_types):
        raw_per_match = [type_count_by_match[mid].get(t, 0) for mid in raw_ids if mid in type_count_by_match]
        elig_per_match = [type_count_by_match[mid].get(t, 0) for mid in raw_ids if mid in eligible_ids]

        raw_count = sum(raw_per_match)
        elig_count = sum(elig_per_match)
        raw_matches_with = sum(1 for v in raw_per_match if v > 0)
        elig_matches_with = sum(1 for v in elig_per_match if v > 0)

        raw_stats = mean_median_std(raw_per_match)
        elig_stats = mean_median_std(elig_per_match)

        rows.append({
            "type_name": t,
            "raw_count": raw_count,
            "raw_pct_of_events": (100 * raw_count / raw_total_events) if raw_total_events else None,
            "eligible_count": elig_count,
            "eligible_pct_of_events": (100 * elig_count / eligible_total_events) if eligible_total_events else None,
            "raw_matches_with": raw_matches_with,
            "raw_matches_without": len(raw_per_match) - raw_matches_with,
            "eligible_matches_with": elig_matches_with,
            "eligible_matches_without": len(elig_per_match) - elig_matches_with,
            "raw_per_match_mean": raw_stats["mean"], "raw_per_match_median": raw_stats["median"],
            "raw_per_match_min": raw_stats["min"], "raw_per_match_max": raw_stats["max"], "raw_per_match_std": raw_stats["std"],
            "eligible_per_match_mean": elig_stats["mean"], "eligible_per_match_median": elig_stats["median"],
            "eligible_per_match_min": elig_stats["min"], "eligible_per_match_max": elig_stats["max"], "eligible_per_match_std": elig_stats["std"],
            "in_eight_classes": t in EIGHT_CLASSES,
            "relevance_note": RELEVANCE_NOTES.get(t, RELEVANCE_FALLBACK),
        })

    rows.sort(key=lambda r: -r["raw_count"])
    return rows, raw_total_events, eligible_total_events


def numeric_summary(st: dict) -> dict:
    if st["count"] == 0:
        return {"count": 0, "mean": None, "min": None, "max": None, "median_approx": None, "approximated": False}
    mean = st["sum"] / st["count"]
    reservoir = st["reservoir"]
    median_approx = statistics.median(reservoir) if reservoir else None
    return {
        "count": st["count"],
        "mean": mean,
        "min": st["min"],
        "max": st["max"],
        "median_approx": median_approx,
        "approximated": st["count"] > RESERVOIR_CAP,
    }


def build_field_inventory(agg_raw: FieldAggregator, agg_elig: FieldAggregator) -> list[dict]:
    all_paths = set(agg_raw.field_types.keys()) | set(agg_elig.field_types.keys())
    rows = []
    for path in sorted(all_paths):
        ftype = agg_raw.field_types.get(path) or agg_elig.field_types.get(path)
        parent = path.rsplit(".", 1)[0] if "." in path else ""
        parent = parent[:-2] if parent.endswith("[]") else parent
        raw_denominator = agg_raw.container_counts.get(parent, agg_raw.container_counts.get("", 0))
        elig_denominator = agg_elig.container_counts.get(parent, agg_elig.container_counts.get("", 0))
        raw_present = agg_raw.field_present.get(path, 0)
        elig_present = agg_elig.field_present.get(path, 0)

        entry = {
            "field_path": path,
            "data_type": ftype,
            "raw_denominator": raw_denominator,
            "raw_present": raw_present,
            "raw_missing": max(raw_denominator - raw_present, 0),
            "eligible_denominator": elig_denominator,
            "eligible_present": elig_present,
            "eligible_missing": max(elig_denominator - elig_present, 0),
            "unique_values_raw": len(agg_raw.field_values.get(path, {})) if ftype in ("categorical", "boolean") else None,
            "unique_values_eligible": len(agg_elig.field_values.get(path, {})) if ftype in ("categorical", "boolean") else None,
        }
        if ftype == "numeric":
            entry["numeric_summary_raw"] = numeric_summary(agg_raw.numeric_stats.get(path, {"count": 0}))
            entry["numeric_summary_eligible"] = numeric_summary(agg_elig.numeric_stats.get(path, {"count": 0}))
        rows.append(entry)
    return rows


def build_category_values(agg_raw: FieldAggregator, agg_elig: FieldAggregator, field_inventory: list[dict]) -> tuple[list[dict], list[dict]]:
    main_rows, overflow_rows = [], []
    for finfo in field_inventory:
        if finfo["data_type"] not in ("categorical", "boolean"):
            continue
        path = finfo["field_path"]
        raw_values = agg_raw.field_values.get(path, Counter())
        elig_values = agg_elig.field_values.get(path, Counter())
        raw_total = finfo["raw_present"]
        elig_total = finfo["eligible_present"]
        all_values = set(raw_values.keys()) | set(elig_values.keys())
        is_high_card = len(all_values) > HIGH_CARDINALITY_THRESHOLD
        target = overflow_rows if is_high_card else main_rows
        raw_match_sets = agg_raw.field_value_matches.get(path, {})
        elig_match_sets = agg_elig.field_value_matches.get(path, {})
        # Tie-break by value string, not just -count: `all_values` is a bare
        # `set`, whose iteration order is hash-seed-randomized per Python
        # process (PYTHONHASHSEED). Without the tie-break, two runs over
        # identical data could order same-count rows differently, breaking
        # the determinism this audit requires.
        for value in sorted(all_values, key=lambda v: (-raw_values.get(v, 0), v)):
            rc = raw_values.get(value, 0)
            ec = elig_values.get(value, 0)
            target.append({
                "field_path": path,
                "value": value,
                "raw_count": rc,
                "raw_pct": (100 * rc / raw_total) if raw_total else None,
                "raw_matches": len(raw_match_sets.get(value, ())),
                "eligible_count": ec,
                "eligible_pct": (100 * ec / elig_total) if elig_total else None,
                "eligible_matches": len(elig_match_sets.get(value, ())),
            })
    return main_rows, overflow_rows


def build_enum_id_consistency(agg_raw: FieldAggregator) -> list[dict]:
    """QC check: does every <container>.id value map to exactly one <container>.name
    and vice versa? Reported as a data-quality finding, not assumed."""
    findings = []
    for container_path, pairs in agg_raw.enum_id_name_map.items():
        id_to_names: dict = defaultdict(set)
        name_to_ids: dict = defaultdict(set)
        for (id_val, name_val), count in pairs.items():
            id_to_names[id_val].add(name_val)
            name_to_ids[name_val].add(id_val)
        bad_ids = {k: v for k, v in id_to_names.items() if len(v) > 1}
        bad_names = {k: v for k, v in name_to_ids.items() if len(v) > 1}
        if bad_ids or bad_names:
            findings.append({
                "container_path": container_path,
                "ids_mapping_to_multiple_names": {str(k): sorted(v) for k, v in bad_ids.items()},
                "names_mapping_to_multiple_ids": {str(k): sorted(v) for k, v in bad_names.items()},
            })
    return findings


def build_goals_audit(audit: dict) -> dict:
    goal_tally = audit["goal_tally"]
    match_meta = audit["match_meta"]
    eligible_ids = set(audit["eligible_ids"])

    mismatches = []
    matches_checked = 0
    matches_matching = 0
    for mid in audit["raw_ids"]:
        meta = match_meta.get(mid)
        if meta is None or meta.get("home_score") is None or meta.get("away_score") is None:
            continue
        if mid not in audit["type_count_by_match"]:
            continue
        matches_checked += 1
        shot_goals = audit["shot_goals_by_match_team"].get(mid, Counter())
        og_credit = audit["own_goal_credit_by_match_team"].get(mid, Counter())
        home_id, away_id = meta.get("home_team_id"), meta.get("away_team_id")
        computed_home = shot_goals.get(home_id, 0) + og_credit.get(home_id, 0)
        computed_away = shot_goals.get(away_id, 0) + og_credit.get(away_id, 0)
        if computed_home == meta["home_score"] and computed_away == meta["away_score"]:
            matches_matching += 1
        else:
            mismatches.append({
                "match_id": mid,
                "eligible": mid in eligible_ids,
                "home_team": meta.get("home_team_name"), "away_team": meta.get("away_team_name"),
                "recorded_home_score": meta["home_score"], "recorded_away_score": meta["away_score"],
                "computed_home_goals": computed_home, "computed_away_goals": computed_away,
                "competition_stage": meta.get("competition_stage"),
            })

    return {
        "shot_goal_raw": goal_tally["shot_goal_raw"],
        "shot_goal_eligible": goal_tally["shot_goal_eligible"],
        "own_goal_for_raw": goal_tally["own_goal_for_raw"],
        "own_goal_for_eligible": goal_tally["own_goal_for_eligible"],
        "own_goal_against_raw": goal_tally["own_goal_against_raw"],
        "own_goal_against_eligible": goal_tally["own_goal_against_eligible"],
        "own_goal_pair_balanced": goal_tally["own_goal_for_raw"] == goal_tally["own_goal_against_raw"],
        "total_goals_raw": goal_tally["shot_goal_raw"] + goal_tally["own_goal_against_raw"],
        "total_goals_eligible": goal_tally["shot_goal_eligible"] + goal_tally["own_goal_against_eligible"],
        "matches_checked_against_metadata": matches_checked,
        "matches_matching_metadata": matches_matching,
        "matches_mismatching_metadata": len(mismatches),
        "mismatch_examples": mismatches[:50],
    }


def build_360_summary(audit: dict) -> dict:
    agg_raw = audit["agg_360_raw"]
    agg_elig = audit["agg_360_eligible"]
    snap_raw = [v for mid, v in audit["snapshot_counts_per_match"].items()]
    snap_elig = [v for mid, v in audit["snapshot_counts_per_match"].items() if mid in set(audit["eligible_ids"])]
    play_raw = [v for mid, v in audit["player_entry_counts_per_match"].items()]
    play_elig = [v for mid, v in audit["player_entry_counts_per_match"].items() if mid in set(audit["eligible_ids"])]

    return {
        "matches_with_360_raw": len(audit["raw_ids"]),
        "matches_with_360_eligible": len(audit["eligible_ids"]),
        "total_snapshots_raw": sum(snap_raw),
        "total_snapshots_eligible": sum(snap_elig),
        "total_freeze_frame_players_raw": sum(play_raw),
        "total_freeze_frame_players_eligible": sum(play_elig),
        "snapshots_per_match_raw": mean_median_std(snap_raw),
        "snapshots_per_match_eligible": mean_median_std(snap_elig),
        "players_per_match_raw": mean_median_std(play_raw),
        "players_per_match_eligible": mean_median_std(play_elig),
        "empty_freeze_frame_raw": agg_raw.list_empty_counts.get("freeze_frame", 0),
        "empty_freeze_frame_eligible": agg_elig.list_empty_counts.get("freeze_frame", 0),
    }


def build_prediction_classes_table(audit: dict, event_type_rows: list[dict]) -> list[dict]:
    eight_by_match = audit["eight_class_current_with_360_by_match"]
    eligible_ids = set(audit["eligible_ids"])
    raw_total = sum(sum(c.values()) for c in eight_by_match.values())
    elig_total = sum(sum(c.values()) for mid, c in eight_by_match.items() if mid in eligible_ids)
    by_type_lookup = {r["type_name"]: r for r in event_type_rows}

    rows = []
    for t in sorted(EIGHT_CLASSES):
        raw_c = sum(c.get(t, 0) for c in eight_by_match.values())
        elig_c = sum(c.get(t, 0) for mid, c in eight_by_match.items() if mid in eligible_ids)
        occurrence_row = by_type_lookup.get(t, {})
        rows.append({
            "class_name": t,
            "pipeline_current_event_count_raw": raw_c,
            "pipeline_current_event_pct_raw": (100 * raw_c / raw_total) if raw_total else None,
            "pipeline_current_event_count_eligible": elig_c,
            "pipeline_current_event_pct_eligible": (100 * elig_c / elig_total) if elig_total else None,
            "all_occurrences_count_raw": occurrence_row.get("raw_count"),
            "all_occurrences_count_eligible": occurrence_row.get("eligible_count"),
        })
    return rows


# ---------------------------------------------------------------------------
# Excel workbook
# ---------------------------------------------------------------------------


def write_excel(results: dict, path: Path) -> None:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo

    wb = Workbook()
    wb.remove(wb.active)
    table_counter = [0]

    def add_sheet(title: str, headers: list[str], rows: list[list], pct_cols: set[int] = frozenset(), int_cols: set[int] = frozenset(), widths: dict[int, int] | None = None, as_table: bool = True, note: str | None = None):
        ws = wb.create_sheet(title[:31])
        start_row = 1
        if note:
            ws.cell(row=1, column=1, value=note).font = Font(italic=True, size=9)
            ws.cell(row=1, column=1).alignment = Alignment(wrap_text=True)
            start_row = 3
        header_row = start_row
        for col_idx, h in enumerate(headers, start=1):
            cell = ws.cell(row=header_row, column=col_idx, value=h)
            cell.font = Font(bold=True)
        for r_idx, row in enumerate(rows, start=header_row + 1):
            for c_idx, value in enumerate(row, start=1):
                if isinstance(value, set):
                    value = ", ".join(map(str, sorted(value)))
                cell = ws.cell(row=r_idx, column=c_idx, value=value)
                if c_idx in pct_cols and isinstance(value, (int, float)):
                    cell.number_format = "0.00"
                elif c_idx in int_cols and isinstance(value, (int, float)):
                    cell.number_format = "#,##0"
        last_row = header_row + len(rows)
        last_col = len(headers)
        ws.freeze_panes = ws.cell(row=header_row + 1, column=1)
        if last_row > header_row:
            ref = f"{get_column_letter(1)}{header_row}:{get_column_letter(last_col)}{last_row}"
            if as_table and last_row > header_row:
                table_counter[0] += 1
                table = Table(displayName=f"Tbl{table_counter[0]}", ref=ref)
                table.tableStyleInfo = TableStyleInfo(name="TableStyleMedium2", showRowStripes=True)
                ws.add_table(table)
            else:
                ws.auto_filter.ref = ref
        for c_idx in range(1, last_col + 1):
            w = (widths or {}).get(c_idx, 18)
            ws.column_dimensions[get_column_letter(c_idx)].width = w
        return ws

    meta = results["meta"]

    # 1. Executive Summary
    exec_rows = [
        ["Audit timestamp (UTC)", meta["timestamp"]],
        ["Raw population: matches with local 360 JSON", meta["raw_match_count"]],
        ["Eligible population (raw minus documented exclusions)", meta["eligible_match_count"]],
        ["Excluded matches (configs/excluded_matches.json)", meta["excluded_match_count"]],
        ["Total StatsBomb event files in local checkout (incl. matches without 360)", meta["total_event_files"]],
        ["Malformed/unreadable files encountered during this audit", meta["malformed_file_count"]],
        ["Total raw events (sum over raw population)", meta["raw_total_events"]],
        ["Total eligible events (sum over eligible population)", meta["eligible_total_events"]],
        ["Unique primary event types (type.name) discovered", meta["unique_event_types"]],
        ["Categorical/boolean fields discovered (full taxonomy)", meta["categorical_field_count"]],
        ["Distinct categorical values discovered (sum across all fields)", meta["distinct_category_value_count"]],
        ["Total 360 snapshots (raw / eligible)", f"{meta['snapshots_raw']} / {meta['snapshots_eligible']}"],
        ["Total 360 freeze-frame player entries (raw / eligible)", f"{meta['players_raw']} / {meta['players_eligible']}"],
        ["Goals via Shot outcome == 'Goal' (raw / eligible)", f"{meta['shot_goals_raw']} / {meta['shot_goals_eligible']}"],
        ["Own goals, i.e. Own Goal Against events (raw / eligible)", f"{meta['own_goals_raw']} / {meta['own_goals_eligible']}"],
        ["Total goals, shot+own (raw / eligible)", f"{meta['total_goals_raw']} / {meta['total_goals_eligible']}"],
        ["Matches reconciled exactly against recorded score", f"{meta['goal_matches_matching']} / {meta['goal_matches_checked']}"],
        ["Current 8-class target implemented as MEANINGFUL_EVENT_TYPES", "Yes (src/build_statsbomb_360_dataset.py)"],
        ["'Goal' is a distinguishable target label today", "No -- see Goals Audit / Prediction Classes"],
    ]
    add_sheet(
        "Executive Summary", ["Metric", "Value"], exec_rows, widths={1: 62, 2: 40}, as_table=False,
        note="All percentages elsewhere in this workbook state their own denominator in an adjacent column or header. Raw = all matches with a local 360 JSON file (426). Eligible = raw minus configs/excluded_matches.json.",
    )

    # 2. Event Types
    et_headers = [
        "Event Type (type.name)", "Raw Count", "Raw % of Raw Events", "Eligible Count", "Eligible % of Eligible Events",
        "Raw Matches With", "Raw Matches Without", "Eligible Matches With", "Eligible Matches Without",
        "Raw/Match Mean", "Raw/Match Median", "Raw/Match Min", "Raw/Match Max", "Raw/Match Std",
        "Eligible/Match Mean", "Eligible/Match Median", "Eligible/Match Min", "Eligible/Match Max", "Eligible/Match Std",
        "In Current 8 Classes", "Thesis Relevance Note",
    ]
    et_rows = [
        [r["type_name"], r["raw_count"], r["raw_pct_of_events"], r["eligible_count"], r["eligible_pct_of_events"],
         r["raw_matches_with"], r["raw_matches_without"], r["eligible_matches_with"], r["eligible_matches_without"],
         r["raw_per_match_mean"], r["raw_per_match_median"], r["raw_per_match_min"], r["raw_per_match_max"], r["raw_per_match_std"],
         r["eligible_per_match_mean"], r["eligible_per_match_median"], r["eligible_per_match_min"], r["eligible_per_match_max"], r["eligible_per_match_std"],
         r["in_eight_classes"], r["relevance_note"]]
        for r in results["event_types"]
    ]
    add_sheet(
        "Event Types", et_headers, et_rows, pct_cols={3, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
        int_cols={2, 4, 6, 7, 8, 9}, widths={1: 18, 21: 70},
        note="Raw % denominator = total raw events across the raw population. Eligible % denominator = total eligible events. Per-match stats include matches with zero occurrences (never silently excluded).",
    )

    # 3. Field Inventory
    fi_headers = ["Field Path", "Data Type", "Raw Denominator", "Raw Present", "Raw Missing", "Eligible Denominator", "Eligible Present", "Eligible Missing", "Unique Values (Raw)", "Unique Values (Eligible)"]
    fi_rows = [
        [f["field_path"], f["data_type"], f["raw_denominator"], f["raw_present"], f["raw_missing"],
         f["eligible_denominator"], f["eligible_present"], f["eligible_missing"], f["unique_values_raw"], f["unique_values_eligible"]]
        for f in results["field_inventory"]
    ]
    add_sheet(
        "Field Inventory", fi_headers, fi_rows, int_cols={3, 4, 5, 6, 7, 8, 9, 10}, widths={1: 48},
        note="Denominator = number of times this field's immediate parent object was itself present (e.g. pass.outcome.name's denominator is the number of Pass events, not all events). Root-level fields use the full event/record count.",
    )

    # 4. Category Values (+ overflow)
    cv_headers = ["Field Path", "Value", "Raw Count", "Raw % (of field's present count)", "Raw Matches", "Eligible Count", "Eligible % (of field's present count)", "Eligible Matches"]
    cv_rows = [[r["field_path"], r["value"], r["raw_count"], r["raw_pct"], r["raw_matches"], r["eligible_count"], r["eligible_pct"], r["eligible_matches"]] for r in results["category_values"]]
    add_sheet(
        "Category Values", cv_headers, cv_rows, pct_cols={4, 7}, int_cols={3, 5, 6, 8}, widths={1: 42, 2: 46},
        note=f"One row per (field, distinct value). Fields with more than {HIGH_CARDINALITY_THRESHOLD} distinct raw values are moved to 'Category Values (High Card)' to keep this sheet navigable -- no values are omitted from the workbook. % denominator = the field's own 'present' count (see Field Inventory), not the whole dataset.",
    )
    if results["category_values_overflow"]:
        cvo_rows = [[r["field_path"], r["value"], r["raw_count"], r["raw_pct"], r["raw_matches"], r["eligible_count"], r["eligible_pct"], r["eligible_matches"]] for r in results["category_values_overflow"]]
        add_sheet(
            "Category Values (High Card)", cv_headers, cvo_rows, pct_cols={4, 7}, int_cols={3, 5, 6, 8}, widths={1: 42, 2: 46},
            note=f"High-cardinality fields (> {HIGH_CARDINALITY_THRESHOLD} distinct raw values, e.g. player names) moved here from 'Category Values' for readability. Same columns, same denominators, nothing truncated.",
        )

    # 5. Event Outcomes
    outcome_suffixes = (".outcome.name", ".technique.name", ".body_part.name", ".card.name", ".type.name", ".height.name")
    outcome_rows_data = [
        r for r in results["category_values"] + results["category_values_overflow"]
        if any(r["field_path"].endswith(s) for s in outcome_suffixes) and r["field_path"] != "type.name"
    ]
    eo_headers = cv_headers
    eo_rows = [[r["field_path"], r["value"], r["raw_count"], r["raw_pct"], r["raw_matches"], r["eligible_count"], r["eligible_pct"], r["eligible_matches"]] for r in sorted(outcome_rows_data, key=lambda r: (r["field_path"], -r["raw_count"]))]
    add_sheet(
        "Event Outcomes", eo_headers, eo_rows, pct_cols={4, 7}, int_cols={3, 5, 6, 8}, widths={1: 42, 2: 46},
        note="Subset of Category Values restricted to *.outcome/type/technique/body_part/card/height.name fields (shots, passes, dribbles, duels, goalkeeper actions, fouls, etc.). type.name itself (primary event types) is reported in the Event Types sheet, not duplicated here.",
    )

    # 6. 360 Categories
    categories_360 = [r for r in results["category_values"] + results["category_values_overflow"] if r["field_path"].startswith("freeze_frame[]") or r["field_path"] in ("visible_area",)]
    s360_headers = cv_headers
    s360_rows = [[r["field_path"], r["value"], r["raw_count"], r["raw_pct"], r["raw_matches"], r["eligible_count"], r["eligible_pct"], r["eligible_matches"]] for r in categories_360]
    s360 = results["summary_360"]
    note_360 = (
        f"Denominator for freeze_frame[].* fields = total freeze-frame player entries (raw {s360['total_freeze_frame_players_raw']:,}, "
        f"eligible {s360['total_freeze_frame_players_eligible']:,}), NOT the number of snapshots. "
        f"Snapshots (raw {s360['total_snapshots_raw']:,}, eligible {s360['total_snapshots_eligible']:,}) are three-sixty.json records; "
        "a player entry is one element of one snapshot's freeze_frame array -- the two must never be conflated."
    )
    add_sheet("360 Categories", s360_headers, s360_rows, pct_cols={4, 7}, int_cols={3, 5, 6, 8}, widths={1: 32, 2: 20}, note=note_360)

    # 7. Goals Audit
    ga = results["goals_audit"]
    ga_rows = [
        ["Shot events with shot.outcome.name == 'Goal' (raw)", ga["shot_goal_raw"]],
        ["Shot events with shot.outcome.name == 'Goal' (eligible)", ga["shot_goal_eligible"]],
        ["'Own Goal For' events (raw)", ga["own_goal_for_raw"]],
        ["'Own Goal For' events (eligible)", ga["own_goal_for_eligible"]],
        ["'Own Goal Against' events (raw)", ga["own_goal_against_raw"]],
        ["'Own Goal Against' events (eligible)", ga["own_goal_against_eligible"]],
        ["Own Goal For/Against pair counts balanced (raw)", ga["own_goal_pair_balanced"]],
        ["Total goals = Shot-Goal + Own Goal Against (raw)", ga["total_goals_raw"]],
        ["Total goals = Shot-Goal + Own Goal Against (eligible)", ga["total_goals_eligible"]],
        ["Matches with usable score metadata checked", ga["matches_checked_against_metadata"]],
        ["Matches where computed goals == recorded home/away score", ga["matches_matching_metadata"]],
        ["Matches where computed goals != recorded score", ga["matches_mismatching_metadata"]],
    ]
    add_sheet("Goals Audit", ["Metric", "Value"], ga_rows, widths={1: 62, 2: 20}, as_table=False,
               note="Own-goal credit for score reconciliation uses 'Own Goal For' (team=beneficiary); own-goal incident count uses 'Own Goal Against' (team=team that conceded, player=who scored into their own net). Mismatches are listed in the Data Quality sheet.")
    if ga["mismatch_examples"]:
        mm_headers = ["Match ID", "Eligible", "Home Team", "Away Team", "Recorded Home Score", "Recorded Away Score", "Computed Home Goals", "Computed Away Goals", "Competition Stage"]
        mm_rows = [[m["match_id"], m["eligible"], m["home_team"], m["away_team"], m["recorded_home_score"], m["recorded_away_score"], m["computed_home_goals"], m["computed_away_goals"], m["competition_stage"]] for m in ga["mismatch_examples"]]
        add_sheet(
            "Goals Audit Mismatches", mm_headers, mm_rows, int_cols={5, 6, 7, 8}, widths={3: 24, 4: 24},
            note="Matches where Shot-Goal + Own-Goal-For credit (computed from events) does not equal the recorded home_score/away_score in match metadata. Common cause: penalty-shootout kicks (period 5) recorded as Shot events but not counted in the regulation/ET score.",
        )

    # 8. Prediction Classes
    pc_headers = ["Class", "Pipeline Current-Event Count (Raw)", "Pipeline Current-Event % (Raw)", "Pipeline Current-Event Count (Eligible)", "Pipeline Current-Event % (Eligible)", "All Occurrences, any 360 status (Raw)", "All Occurrences, any 360 status (Eligible)"]
    pc_rows = [[r["class_name"], r["pipeline_current_event_count_raw"], r["pipeline_current_event_pct_raw"], r["pipeline_current_event_count_eligible"], r["pipeline_current_event_pct_eligible"], r["all_occurrences_count_raw"], r["all_occurrences_count_eligible"]] for r in results["prediction_classes"]]
    add_sheet(
        "Prediction Classes", pc_headers, pc_rows, pct_cols={3, 5}, int_cols={2, 4, 6, 7}, widths={1: 18},
        note="'Pipeline current-event count' requires the event to ALSO have 360 coverage (current_event_id in frames_by_event_id), matching src/build_statsbomb_360_dataset.py exactly. 'All occurrences' is the type's raw count regardless of 360 coverage (from the Event Types sheet) -- the gap between the two columns is 360-coverage loss, not a labeling difference.",
    )

    # 9. Match Statistics
    ms_headers = ["Match ID", "Competition", "Season", "Home Team", "Away Team", "Home Score", "Away Score", "Match Date", "Eligible", "Exclusion Reason", "Load Status", "Total Events", "Total 360 Snapshots", "Total Freeze-Frame Players"]
    ms_rows = [
        [r["match_id"], r.get("competition"), r.get("season"), r.get("home_team_name"), r.get("away_team_name"), r.get("home_score"), r.get("away_score"), r.get("match_date"), r["eligible"], r["exclusion_reason"], r["load_status"], r["total_events"], r["total_three_sixty_records"], r["total_freeze_frame_players"]]
        for r in results["match_rows"]
    ]
    add_sheet("Match Statistics", ms_headers, ms_rows, int_cols={6, 7, 12, 13, 14}, widths={2: 22, 4: 22, 5: 22})

    # 10. Data Quality
    dq_rows: list[list] = [["Malformed/unreadable files", len(results["malformed_files"])]]
    for mf in results["malformed_files"]:
        dq_rows.append([f"  match {mf['match_id']}: {mf['error_type']}: {mf['error_message']} ({mf['file']})", ""])
    dq_rows.append(["Enum id/name consistency violations (container.id mapping to >1 container.name or vice versa)", len(results["enum_id_consistency"])])
    for finding in results["enum_id_consistency"]:
        dq_rows.append([f"  {finding['container_path']}: bad ids={finding['ids_mapping_to_multiple_names']}; bad names={finding['names_mapping_to_multiple_ids']}", ""])
    dq_rows.append(["Own-goal For/Against pair balance (raw)", results["goals_audit"]["own_goal_pair_balanced"]])
    dq_rows.append(["Score-reconciliation mismatches (see Goals Audit)", results["goals_audit"]["matches_mismatching_metadata"]])
    dq_rows.append(["Total local event files without any 360 data (out of audit scope)", results["meta"]["total_event_files"] - results["meta"]["raw_match_count"]])
    dq_rows.append(["player.position.name occurrences (illustrative task spec field)", results["field_lookup"].get("player.position.name", 0)])
    dq_rows.append(["position.name occurrences (actual schema: top-level sibling of player, not nested)", results["field_lookup"].get("position.name", 0)])
    add_sheet("Data Quality", ["Finding", "Value"], dq_rows, widths={1: 90, 2: 20}, as_table=False)

    # 11. Recommendations
    rec_rows = [
        ["A. Current 8-class classification", "8", "High (top 2 classes ~88%)", "Low implementation risk; well-documented baseline; cannot represent Goal, Ball Receipt*, or defensive actions.", "Recommended as primary task; see note below."],
        ["B. All observed primary event types", f"{meta['unique_event_types']}", "Very high (many single-digit-% or <1% classes)", "Administrative/stoppage types add noise without football meaning as a 'next event'.", "Not recommended as-is."],
        ["C. Broader on-ball event set (documented inclusion rule)", "~12-16 (8 current + Ball Receipt*, Duel, Interception, Block, 50/50, Goal Keeper)", "Moderate-high; new classes are mostly rare", "Materially improves coverage of on-ball actions at a manageable added-class cost.", "Recommended as a secondary/extended experiment."],
        ["D. Hierarchical: event type, then outcome (e.g. Shot -> Goal/No Goal)", "8 (+ binary/few-way outcome heads)", "Outcome heads highly imbalanced (Goal is a small minority of Shots)", "Directly addresses the 'cannot distinguish Goal' gap without exploding the primary class count.", "Recommended as a complementary head alongside A."],
    ]
    rec_headers = ["Alternative", "Number of Classes", "Class Imbalance", "Key Tradeoff", "Recommendation"]
    add_sheet(
        "Recommendations", rec_headers, rec_rows, widths={1: 46, 4: 60, 5: 40}, as_table=False,
        note="Full narrative comparison and the data-supported reasoning behind each recommendation is in docs/statsbomb_complete_category_audit.md section 7. This sheet is a compact summary, not a replacement for that discussion.",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    wb.save(path)


# ---------------------------------------------------------------------------
# JSON-serialisable summary (also used by the Excel writer and the report)
# ---------------------------------------------------------------------------


def to_jsonable(obj):
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Counter):
        return dict(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample", type=int, default=None, help="Audit a random sample of N matches instead of all 426 (for fast iteration).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    log("Loading exclusion manifest and discovering match population...")
    audit = run_audit(args.sample, args.seed)

    log("Building event-type table...")
    event_type_rows, raw_total_events, eligible_total_events = build_event_type_table(audit)

    log("Building field inventory (raw + eligible)...")
    field_inventory = build_field_inventory(audit["agg_events_raw"], audit["agg_events_eligible"])
    field_inventory_360 = build_field_inventory(audit["agg_360_raw"], audit["agg_360_eligible"])
    # Namespace the 360 field paths so they never collide with event field paths.
    for f in field_inventory_360:
        f["field_path"] = f["field_path"]
    full_field_inventory = field_inventory + field_inventory_360

    log("Building category-value tables...")
    cat_values, cat_values_overflow = build_category_values(audit["agg_events_raw"], audit["agg_events_eligible"], field_inventory)
    cat_values_360, cat_values_360_overflow = build_category_values(audit["agg_360_raw"], audit["agg_360_eligible"], field_inventory_360)
    cat_values += cat_values_360
    cat_values_overflow += cat_values_360_overflow

    log("Checking enum id/name consistency...")
    enum_findings = build_enum_id_consistency(audit["agg_events_raw"]) + build_enum_id_consistency(audit["agg_360_raw"])

    log("Building goals audit...")
    goals_audit = build_goals_audit(audit)

    log("Building 360 summary...")
    summary_360 = build_360_summary(audit)

    log("Building prediction-classes table...")
    prediction_classes = build_prediction_classes_table(audit, event_type_rows)

    field_lookup = {f["field_path"]: f["raw_present"] for f in full_field_inventory}

    distinct_category_value_count = len(cat_values) + len(cat_values_overflow)
    categorical_field_count = sum(1 for f in full_field_inventory if f["data_type"] in ("categorical", "boolean"))

    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_match_count": len(audit["raw_ids"]),
        "eligible_match_count": len(audit["eligible_ids"]),
        "excluded_match_count": len(audit["excluded_ids"] & set(audit["raw_ids"])),
        "total_event_files": audit["total_event_files"],
        "malformed_file_count": len(audit["malformed_files"]),
        "raw_total_events": raw_total_events,
        "eligible_total_events": eligible_total_events,
        "unique_event_types": len(event_type_rows),
        "categorical_field_count": categorical_field_count,
        "distinct_category_value_count": distinct_category_value_count,
        "snapshots_raw": summary_360["total_snapshots_raw"],
        "snapshots_eligible": summary_360["total_snapshots_eligible"],
        "players_raw": summary_360["total_freeze_frame_players_raw"],
        "players_eligible": summary_360["total_freeze_frame_players_eligible"],
        "shot_goals_raw": goals_audit["shot_goal_raw"],
        "shot_goals_eligible": goals_audit["shot_goal_eligible"],
        "own_goals_raw": goals_audit["own_goal_against_raw"],
        "own_goals_eligible": goals_audit["own_goal_against_eligible"],
        "total_goals_raw": goals_audit["total_goals_raw"],
        "total_goals_eligible": goals_audit["total_goals_eligible"],
        "goal_matches_matching": goals_audit["matches_matching_metadata"],
        "goal_matches_checked": goals_audit["matches_checked_against_metadata"],
    }

    results = {
        "meta": meta,
        "event_types": event_type_rows,
        "field_inventory": full_field_inventory,
        "category_values": cat_values,
        "category_values_overflow": cat_values_overflow,
        "enum_id_consistency": enum_findings,
        "goals_audit": goals_audit,
        "summary_360": summary_360,
        "prediction_classes": prediction_classes,
        "match_rows": audit["match_rows"],
        "malformed_files": audit["malformed_files"],
        "field_lookup": field_lookup,
    }

    log("Writing JSON intermediate results...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=to_jsonable)

    log("Writing Excel workbook...")
    write_excel(results, OUTPUT_XLSX)

    log("Done.")
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"\nJSON: {OUTPUT_JSON}")
    print(f"XLSX: {OUTPUT_XLSX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
