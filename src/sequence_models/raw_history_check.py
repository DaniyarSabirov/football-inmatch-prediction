"""Independent reconstruction of (history, target) from raw StatsBomb event JSON (experiment 5).

It shares no code with ``tasks.build_task_table``, ``v3_taxonomy.companion_rules`` or
``event_history``. It re-implements the canonical sequence directly on the raw records:

- records of one period, in raw ``index`` order (period 5 never occurs for a sampled row);
- keep records whose ``type.name`` is one of the 15 classes;
- drop companions, read from the raw JSON keys:
  K1 ``goalkeeper.type`` in {Shot Faced, Goal Conceded, Penalty Conceded};
  K2 ``duel.type`` == Tackle, previous raw record Dispossessed or Dribble with
  ``dribble.outcome`` == Incomplete, identical ``timestamp`` string, previous id in
  ``related_events``;
  K3 50/50 whose previous raw record is a 50/50 with identical timestamp and linked.

It is used to verify a sample of real feature vectors and targets."""

from __future__ import annotations

import json
from pathlib import Path

from .canonical_task import CLASSES
from .event_history import PAD

GK_NO_ACTION = {"Shot Faced", "Goal Conceded", "Penalty Conceded"}
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}


def _name(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def is_companion(ev: dict, prev: dict | None) -> bool:
    typ = _name(ev, "type", "name")
    if typ == "Goal Keeper" and _name(ev, "goalkeeper", "type", "name") in GK_NO_ACTION:
        return True
    if prev is None:
        return False
    linked = prev["id"] in (ev.get("related_events") or [])
    same_ts = prev.get("timestamp") == ev.get("timestamp")
    ptyp = _name(prev, "type", "name")
    if typ == "Duel" and _name(ev, "duel", "type", "name") == "Tackle" and linked and same_ts:
        if ptyp == "Dispossessed" or (ptyp == "Dribble" and _name(prev, "dribble", "outcome", "name") == "Incomplete"):
            return True
    if typ == "50/50" and ptyp == "50/50" and linked and same_ts:
        return True
    return False


def canonical_stream(events: list[dict], period: int) -> list[dict]:
    """Canonical observations of one period from raw records."""
    raw = sorted((e for e in events if e.get("period") == period), key=lambda e: e["index"])
    out, prev = [], None
    for ev in raw:
        if _name(ev, "type", "name") in CLASS_ID and not is_companion(ev, prev):
            out.append(ev)
        prev = ev
    return out


def reconstruct(events: list[dict], event_id: str, n_previous: int = 4) -> tuple[list[int], int | None, list[str]]:
    """(history ids E_{t-n}..E_t with PAD, target id or None, history event ids) for one record."""
    ev = next(e for e in events if e["id"] == event_id)
    stream = canonical_stream(events, ev["period"])
    pos = next(i for i, e in enumerate(stream) if e["id"] == event_id)
    hist = []
    for lag in range(n_previous, -1, -1):
        hist.append(CLASS_ID[stream[pos - lag]["type"]["name"]] if pos - lag >= 0 else PAD)
    target = CLASS_ID[stream[pos + 1]["type"]["name"]] if pos + 1 < len(stream) else None
    ids = [stream[pos - lag]["id"] if pos - lag >= 0 else "" for lag in range(n_previous, -1, -1)]
    return hist, target, ids


def load_events(events_dir: Path, match_id: int) -> list[dict]:
    return json.loads((events_dir / f"{match_id}.json").read_text(encoding="utf-8"))
