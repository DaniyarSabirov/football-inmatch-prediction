r"""Canonical prediction task ``actions15_360current_v1`` (approved 2026-11-19).

Specification: ``docs/canonical_prediction_protocol.md``. Declarative copy:
``configs/actions15_360current_v1.yaml`` (validated against the constants below).

Definition. Let the canonical sequence of one (match, period) be the raw StatsBomb records,
in raw ``index`` order, whose type is one of the K = 15 ``actions15`` classes and that are not
companion records (rules K1-K3, ``v3_taxonomy.companion_rules``); period 5 is excluded. For
an observation :math:`E_t` of this sequence the target is the class of the next observation
:math:`E_{t+1}` of the same sequence. A *row* exists for every :math:`E_t` that has a target
and a usable 360 snapshot of its own (a three-sixty record with a non-empty freeze frame).
The same row rule applies to training and evaluation (Protocol B). The target event needs no
snapshot. Features may use only information available at the observation time of
:math:`E_t` (its type, its own snapshot, earlier records).

Nothing here changes earlier task versions: ``scenarioD15`` (experiments 2-4) and the
production 8-class dataset are untouched.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.paths import PROJECT_ROOT

from .config import TaskSpec
from .event_classes import SCENARIO_D15_CLASSES
from .tasks import NO_TARGET, build_task_table, validate_event_order
from .v3_taxonomy import COMPANION, D15, companion_rules

TASK_VERSION = "actions15_360current_v1"
CONFIG_PATH = PROJECT_ROOT / "configs" / f"{TASK_VERSION}.yaml"

#: Canonical class order; class id = position. Same names and order as experiments 2-4.
CLASSES: tuple[str, ...] = SCENARIO_D15_CLASSES
K = len(CLASSES)
COMPANION_RULES = ("K1", "K2", "K3")
EXCLUDED_PERIODS = (5,)

#: Raw StatsBomb fields that describe what happens AFTER the observation time of the current
#: event (or point to later records). They are never features of the canonical task.
FORBIDDEN_RAW_FIELDS = frozenset({
    "outcome", "end_location", "end_x", "end_y", "recipient", "recipient_id", "related_events",
    "duration", "aerial_won", "pass_outcome", "shot_outcome", "duel_outcome",
    "target", "target_id", "target_event_id", "target_raw_type", "target_has_360",
    "n_skipped_raw_events", "time_to_next_event_seconds", "possession", "possession_team_id",
})

#: Columns of the canonical table (the only columns it exposes).
TABLE_COLUMNS = ["task_version", "match_id", "period", "sequence_id", "seq_pos", "index", "list_position",
                 "event_id", "symbol", "symbol_id", "has_360", "usable_360", "target", "target_id",
                 "target_event_id", "target_has_360", "n_skipped_raw_events", "row_eligible"]

if tuple(D15) != CLASSES:  # pragma: no cover - guarded by tests as well
    raise RuntimeError("actions15 class order differs from the canonical class order")


class CanonicalTaskError(ValueError):
    """The configuration or the data does not satisfy the canonical task definition."""


@dataclass(frozen=True)
class CanonicalTaskSpec:
    task_version: str
    classes: tuple[str, ...]
    companion_rules: tuple[str, ...]
    k4_aerial_lost_rule: bool
    excluded_periods: tuple[int, ...]
    current_event_requires_usable_360: bool
    target_event_requires_360: bool
    raw: dict

    def path(self, key: str) -> Path:
        section = "partition" if key in self.raw["partition"] else "sources"
        return PROJECT_ROOT / self.raw[section][key]


def load_task_spec(path: Path = CONFIG_PATH) -> CanonicalTaskSpec:
    """Load the YAML and refuse any rule that differs from this implementation."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "task_version": TASK_VERSION, "taxonomy": "actions15", "classes": list(CLASSES),
        "companion_rules": list(COMPANION_RULES), "k4_aerial_lost_rule": False,
        "excluded_periods": list(EXCLUDED_PERIODS), "event_order": "statsbomb_index",
        "current_event_requires_usable_360": True, "target_event_requires_360": False,
        "prediction_time": "observation_time_of_current_event",
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise CanonicalTaskError(f"{path}: {key} = {raw.get(key)!r}, but {TASK_VERSION} requires {value!r}")
    return CanonicalTaskSpec(TASK_VERSION, CLASSES, COMPANION_RULES, False, EXCLUDED_PERIODS, True, False, raw)


def task_spec_for_builder() -> TaskSpec:
    return TaskSpec(name=TASK_VERSION, description=TASK_VERSION, vocabulary=CLASSES, sequence_events="vocabulary_only",
                    out_of_vocabulary_label=None, prediction_point_types=CLASSES, excluded_periods=EXCLUDED_PERIODS)


REQUIRED_RAW_COLUMNS = ("match_id", "list_position", "index", "event_id", "period", "timestamp", "t_seconds", "type",
                        "has_360", "has_freeze_frame", "subtype", "outcome", "related_events")


def build_canonical_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Canonical task table from a raw event table with event details.

    ``raw``: one row per raw StatsBomb record (any set of WHOLE matches), with the columns of
    ``raw_events.EVENT_TABLE_COLUMNS`` plus ``subtype``, ``outcome``, ``related_events``
    (``v3_events``). Returns one row per canonical observation (columns ``TABLE_COLUMNS``).
    """
    missing = [c for c in REQUIRED_RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise CanonicalTaskError(f"raw table lacks columns {missing}")
    raw = raw.sort_values(["match_id", "index"], kind="stable").reset_index(drop=True)
    validate_event_order(raw)  # index == file position + 1, unique ids, periods never decrease
    rules = companion_rules(raw)
    relabelled = raw.copy()
    relabelled.loc[(rules != "").to_numpy(), "type"] = COMPANION   # skipped like any non-vocabulary record
    table = build_task_table(relabelled, task_spec_for_builder())
    usable = raw.set_index("event_id")
    usable = usable["has_360"].astype(bool) & usable["has_freeze_frame"].astype(bool)
    table["usable_360"] = table["event_id"].map(usable).astype(bool).to_numpy()
    table["target_has_360"] = table["target_event_id"].map(usable).astype("boolean")
    has_target = table["target_id"].to_numpy() != NO_TARGET
    table["row_eligible"] = has_target & table["usable_360"].to_numpy()   # training AND evaluation rows
    table["task_version"] = TASK_VERSION
    table = table[TABLE_COLUMNS].copy()
    table.attrs["companion_rule_counts"] = rules[(rules != "") & ~raw["period"].isin(EXCLUDED_PERIODS)].value_counts().to_dict()
    check_canonical_table(table)
    return table


def check_canonical_table(table: pd.DataFrame) -> None:
    """Invariants of the canonical table (raise CanonicalTaskError on violation)."""
    if set(table.columns) & FORBIDDEN_RAW_FIELDS - {"target", "target_id", "target_event_id", "target_has_360",
                                                     "n_skipped_raw_events"}:
        raise CanonicalTaskError("canonical table exposes a forbidden raw field")
    if not table["symbol"].isin(CLASSES).all():
        raise CanonicalTaskError("observation outside the 15 classes")
    if (table["symbol_id"].to_numpy() != table["symbol"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()).any():
        raise CanonicalTaskError("class id does not match the canonical order")
    if table["period"].isin(EXCLUDED_PERIODS).any():
        raise CanonicalTaskError("penalty shoot-out observation")
    seq = table["sequence_id"].to_numpy()
    same = seq[1:] == seq[:-1]
    nxt = np.where(same, table["symbol_id"].to_numpy()[1:], NO_TARGET)
    if not np.array_equal(table["target_id"].to_numpy()[:-1], nxt) or (len(table) and table["target_id"].iat[-1] != NO_TARGET):
        raise CanonicalTaskError("target is not the next observation of the same sequence")
    idx = table["index"].to_numpy()
    if (np.diff(idx)[same] <= 0).any():
        raise CanonicalTaskError("observations are not in increasing StatsBomb index order")
    if (table.groupby("sequence_id")[["match_id", "period"]].nunique() > 1).any().any():
        raise CanonicalTaskError("a sequence crosses a match or period")
    if (table["row_eligible"] & ~table["usable_360"]).any():
        raise CanonicalTaskError("eligible row without a usable current-event snapshot")


def dataset_sha256(table: pd.DataFrame) -> str:
    """Deterministic hash of the canonical table content (row order and values)."""
    cols = ["match_id", "period", "index", "event_id", "symbol_id", "target_id", "usable_360", "row_eligible"]
    payload = table[cols].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_development_raw(spec: CanonicalTaskSpec) -> tuple[pd.DataFrame, dict]:
    """Raw events + details of the development matches (Tier A + Tier B). Final-test matches
    are refused by every loader. Returns (raw table, partition info)."""
    import json

    from .config_v2 import load_config_v2
    from .experiment_v2 import load_development_events, load_partition
    from .splits import assert_no_final_test
    from .v3_events import build_event_details

    part = load_partition(load_config_v2())
    if spec.path("split_manifest") != PROJECT_ROOT / "results/markov_hmm_pilot/split_manifest.json":
        raise CanonicalTaskError("the canonical task uses experiment 1's split manifest")
    events = load_development_events(load_config_v2(), part, part.development)
    details = build_event_details(part.development, part.final_test, spec.path("event_details_cache"))
    raw = events.merge(details.drop(columns="match_id"), on="event_id", how="left", validate="one_to_one")
    assert_no_final_test(raw["match_id"].unique().tolist(), part.final_test)
    folds = json.loads(spec.path("folds_file").read_text(encoding="utf-8"))
    return raw, {"partition": part, "folds_payload": folds}
