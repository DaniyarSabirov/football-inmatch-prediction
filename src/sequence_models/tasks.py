"""Prediction-task construction: sequences, targets and evaluation rows.

Definitions (see results/markov_hmm_pilot/README.md, "Task specification"):

- **Sequence**: one (match, period) pair, periods in ``excluded_periods``
  removed entirely. Order is the raw event order (``index``), which the
  audit verified equals the raw array order; timestamps are never used to
  order events (they tie for ~31% of adjacent raw pairs and run backwards
  for 43 events). A sequence never crosses a period or match boundary.
- **Observation**: an event of the sequence, encoded as a vocabulary symbol.
- **Target** at observation t: the symbol of observation t+1 of the same
  sequence. The last observation of a sequence has no target.
- **Prediction point**: an observation whose raw type is a production type
  and that has a target. **Evaluation row**: a prediction point whose
  current event has a 360 record (the production dataset population).

Nothing is silently dropped and reconnected: events that are not in a
sequence are either outside the task's definition (``vocabulary_only``
tasks skip non-vocabulary types by definition, counted per row as
``n_skipped_raw_events``) or belong to an excluded period, which removes the
whole period and so creates no transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TaskSpec

NO_TARGET = -1

REASON_NO_TARGET = "no_target_in_period"
REASON_NO_360 = "current_event_without_360_record"
REASON_NOT_PREDICTION_POINT = "current_event_type_not_a_prediction_point"
REASON_EXCLUDED_PERIOD = "excluded_period_penalty_shootout"
REASON_SKIPPED_BY_DEFINITION = "skipped_by_filtered_task_definition"


class EventOrderError(ValueError):
    """The raw event stream is not in a verifiable order."""


class UnknownEventTypeError(ValueError):
    """An event type cannot be encoded in the task vocabulary."""


def validate_event_order(event_table: pd.DataFrame) -> None:
    """Raise unless, per match, ``index`` equals the raw array position + 1,
    event ids are unique, and periods never decrease along the stream."""
    if event_table["event_id"].duplicated().any():
        raise EventOrderError("duplicate event ids")
    for match_id, group in event_table.groupby("match_id", sort=False):
        positions = group["list_position"].to_numpy()
        if not np.array_equal(positions, np.arange(len(group))):
            raise EventOrderError(f"match {match_id}: list positions are not 0..n-1 in table order")
        if not np.array_equal(group["index"].to_numpy(), positions + 1):
            raise EventOrderError(f"match {match_id}: index differs from raw array order")
        if (np.diff(group["period"].to_numpy()) < 0).any():
            raise EventOrderError(f"match {match_id}: period decreases along the event stream")


def encode_symbol(raw_type: str, task: TaskSpec) -> str:
    """Map a raw event type to the task vocabulary, or raise for an unknown type."""
    if raw_type in task.prediction_point_types:
        return raw_type
    if task.sequence_events == "all_events" and task.out_of_vocabulary_label is not None:
        return task.out_of_vocabulary_label
    raise UnknownEventTypeError(f"{raw_type!r} is not in the vocabulary of task {task.name}")


def build_task_table(event_table: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    """One row per sequence observation, with targets and eligibility.

    ``event_table`` is the flat raw table from ``raw_events`` (any subset of
    whole matches). Returns columns:

    match_id, period, sequence_id, seq_pos, list_position, index, event_id,
    raw_type, symbol, symbol_id, has_360, timestamp, t_seconds,
    target, target_id, target_event_id, target_raw_type,
    n_skipped_raw_events, is_prediction_point, eval_eligible, exclusion_reason
    """
    validate_event_order(event_table)
    table = event_table[~event_table["period"].isin(task.excluded_periods)]
    if task.sequence_events == "vocabulary_only":
        table = table[table["type"].isin(task.vocabulary)]
    table = table.copy()
    symbol_index = {symbol: i for i, symbol in enumerate(task.vocabulary)}
    table["raw_type"] = table["type"]
    table["symbol"] = [encode_symbol(t, task) for t in table["type"]]
    table["symbol_id"] = table["symbol"].map(symbol_index).astype(np.int64)
    table["sequence_id"] = table["match_id"].astype(str) + "_P" + table["period"].astype(str)
    grouped = table.groupby(["match_id", "period"], sort=False)
    table["seq_pos"] = grouped.cumcount()
    table["target"] = grouped["symbol"].shift(-1)
    table["target_id"] = grouped["symbol_id"].shift(-1).fillna(NO_TARGET).astype(np.int64)
    table["target_event_id"] = grouped["event_id"].shift(-1)
    table["target_raw_type"] = grouped["type"].shift(-1)
    next_position = grouped["list_position"].shift(-1)
    table["n_skipped_raw_events"] = (next_position - table["list_position"] - 1).astype("Int64")
    has_target = table["target_id"] != NO_TARGET
    point_type = table["raw_type"].isin(task.prediction_point_types)
    table["is_prediction_point"] = point_type & has_target
    table["eval_eligible"] = table["is_prediction_point"] & table["has_360"]
    reason = np.where(
        ~has_target, REASON_NO_TARGET,
        np.where(~point_type, REASON_NOT_PREDICTION_POINT, np.where(~table["has_360"], REASON_NO_360, "")),
    )
    table["exclusion_reason"] = reason
    columns = [
        "match_id", "period", "sequence_id", "seq_pos", "list_position", "index", "event_id",
        "raw_type", "symbol", "symbol_id", "has_360", "timestamp", "t_seconds",
        "target", "target_id", "target_event_id", "target_raw_type",
        "n_skipped_raw_events", "is_prediction_point", "eval_eligible", "exclusion_reason",
    ]
    return table[columns].reset_index(drop=True)


def sequences(task_table: pd.DataFrame) -> list[np.ndarray]:
    """Row positions of each sequence, in table order (one array per sequence)."""
    codes = task_table["sequence_id"].to_numpy()
    if len(codes) == 0:
        return []
    boundaries = np.flatnonzero(codes[1:] != codes[:-1]) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes)]])
    if len(set(codes[starts])) != len(starts):
        raise EventOrderError("a sequence is split into non-contiguous blocks")
    return [np.arange(s, e) for s, e in zip(starts, ends)]


def exclusion_summary(event_table: pd.DataFrame, task: TaskSpec) -> pd.DataFrame:
    """Event-level accounting of every raw event for one task: where it goes and why.

    Rows: (task, stage, reason, raw_type, n_events, n_matches).
    """
    records = []
    in_excluded_period = event_table["period"].isin(task.excluded_periods)
    for raw_type, group in event_table[in_excluded_period].groupby("type"):
        records.append((REASON_EXCLUDED_PERIOD, raw_type, len(group), group["match_id"].nunique()))
    kept = event_table[~in_excluded_period]
    if task.sequence_events == "vocabulary_only":
        skipped = kept[~kept["type"].isin(task.vocabulary)]
        for raw_type, group in skipped.groupby("type"):
            records.append((REASON_SKIPPED_BY_DEFINITION, raw_type, len(group), group["match_id"].nunique()))
    task_table = build_task_table(event_table, task)
    for (reason, raw_type), group in task_table.groupby(["exclusion_reason", "raw_type"]):
        records.append((reason or "evaluated", raw_type, len(group), group["match_id"].nunique()))
    frame = pd.DataFrame(records, columns=["reason", "raw_type", "n_events", "n_matches"])
    frame.insert(0, "task", task.name)
    return frame.sort_values(["reason", "n_events"], ascending=[True, False]).reset_index(drop=True)
