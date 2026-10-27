"""Diagnostic (investigation only): how do near-simultaneous 360 records of
OPPOSING teams relate to each other?

Per the StatsBomb 360 spec (data/raw/statsbomb/doc, "Open Data 360 Frames
v1.0.0"), every record is event-relative: `location` is oriented with the
event's (actor's) team attacking 0->120 and `teammate` means "same team as
the actor in this event". So two records of the SAME physical moment owned by
opposing teams should be MIRRORED in coordinates and INVERTED in teammate.

For every pair of records owned by different teams in the same period with
|dt| <= threshold (timestamps parsed to integer milliseconds), the players
of both RAW records are paired by optimal assignment (Hungarian) twice: as
stored and with one side mirrored through the pitch centre. The better
hypothesis gives the coordinate relation; the flags of the paired players
give the flag relation:

  coordinate relation  IDENTICAL (max pair distance < 0.01 yd)
                       NEAR (mean < 0.75 yd and max < 2 yd)
                       MIRROR_IDENTICAL / MIRROR_NEAR (same, after mirroring)
                       DIFFERENT
  flag relation        EQUAL (>= 90% of pairs same teammate value)
                       INVERTED (>= 90% opposite)   MIXED otherwise

Spec-consistent pair: MIRROR_* + INVERTED. A "copied" pair is IDENTICAL/NEAR
+ EQUAL: one record is the other team's snapshot (coordinates AND teammate
flags in the other team's perspective), typically with only `actor` moved.

Read-only; nothing in production is used for the classification except the
raw JSON loaders.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"


def ms(timestamp: str) -> int:
    """'HH:MM:SS.mmm' -> integer milliseconds (exact; no float rounding)."""
    hours, minutes, rest = timestamp.split(":")
    seconds, _, fraction = rest.partition(".")
    return ((int(hours) * 60 + int(minutes)) * 60 + int(seconds)) * 1000 + int((fraction + "000")[:3])


def load(match_id: int):
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    joined = [e for e in events if e["id"] in frames and frames[e["id"]].get("freeze_frame")]
    joined.sort(key=lambda e: (e["period"], ms(e["timestamp"]), e["index"]))
    return events, frames, joined


@dataclass(frozen=True)
class Relation:
    coordinates: str
    flags: str
    pairs: int
    mean_distance: float
    max_distance: float
    equal_flags: int
    mapping: tuple[tuple[int, int], ...]  # (index in A, index in B) pairs under the chosen hypothesis
    mirrored: bool


def relate(frame_a: dict, frame_b: dict) -> Relation:
    a = [tuple(p["location"]) for p in frame_a["freeze_frame"]]
    b = [tuple(p["location"]) for p in frame_b["freeze_frame"]]
    best = None
    for mirrored in (False, True):
        bb = [(120 - x, 80 - y) if mirrored else (x, y) for x, y in b]
        cost = [[math.dist(p, q) for q in bb] for p in a]
        rows, cols = linear_sum_assignment(cost)
        distances = [cost[r][c] for r, c in zip(rows, cols)]
        mean = sum(distances) / len(distances)
        if best is None or mean < best[0]:
            best = (mean, max(distances), list(zip(rows, cols)), mirrored)
    mean, worst, mapping, mirrored = best
    if worst < 0.01:
        coordinates = "IDENTICAL"
    elif mean < 0.75 and worst < 2.0:
        coordinates = "NEAR"
    else:
        coordinates = "DIFFERENT"
    if coordinates != "DIFFERENT" and mirrored:
        coordinates = "MIRROR_" + coordinates
    equal = sum(frame_a["freeze_frame"][r]["teammate"] == frame_b["freeze_frame"][c]["teammate"] for r, c in mapping)
    n = len(mapping)
    flags = "EQUAL" if equal >= 0.9 * n else "INVERTED" if n - equal >= 0.9 * n else "MIXED"
    return Relation(coordinates, flags, n, round(mean, 3), round(worst, 3), equal,
                    tuple((int(r), int(c)) for r, c in mapping), mirrored)


def cross_team_pairs(joined: list[dict], threshold_ms: int):
    """Yield (event_a, event_b, dt_ms) for opposing-team records within threshold."""
    for i, a in enumerate(joined):
        for b in joined[i + 1:]:
            if b["period"] != a["period"]:
                break
            dt = ms(b["timestamp"]) - ms(a["timestamp"])
            if dt > threshold_ms:
                break
            if a["team"]["id"] != b["team"]["id"]:
                yield a, b, dt


def same_team_pairs(joined: list[dict], threshold_ms: int):
    for i, a in enumerate(joined):
        for b in joined[i + 1:]:
            if b["period"] != a["period"]:
                break
            dt = ms(b["timestamp"]) - ms(a["timestamp"])
            if dt > threshold_ms:
                break
            if a["team"]["id"] == b["team"]["id"]:
                yield a, b, dt
