r"""Feature-set registry for controlled feature expansion (F0, F1, ...).

A feature set maps the information available at the current event :math:`E_t` to a feature
vector :math:`\mathbf{x}_t \in \mathbb{R}^d`. Each set declares the task-table columns it
may read (``input_columns``). :func:`design_matrix` hands the builder ONLY those columns,
so a builder cannot read the target, identifiers or any other column by accident.

Leakage rule (``configs/ml_feature_classification.json``): a feature may depend only on
the current event, its own 360 snapshot and records EARLIER in the event stream. Columns
derived from the next event are listed in :data:`FORBIDDEN_COLUMNS` and can never be
declared as inputs. Note that ``n_skipped_raw_events`` is target-derived: it counts the
raw records between :math:`E_t` and :math:`E_{t+1}`, so it depends on where the next event
is.

Implemented:

- **F0** current event only: :math:`\mathbf{x}_t = \operatorname{OneHot}(E_t)`, d = 15.

Planned (registered with their intended inputs; building them raises
``NotImplementedError`` until each is implemented and audited): F1 spatial zone, F2 event
coordinates, F3 temporal context, F4 player geometry (360), F5 event history, F6 match
context. See ``docs/thesis/04_feature_engineering_roadmap.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .event_classes import K, SCENARIO_D15_CLASSES, assert_valid_ids

#: Task-table columns that are derived from the next event, or are identifiers / row
#: bookkeeping that would let a model memorise matches. Never a model input.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset({
    "target", "target_id", "target_event_id", "target_raw_type", "n_skipped_raw_events",
    "is_prediction_point", "eval_eligible", "exclusion_reason",
    "match_id", "event_id", "sequence_id", "index", "list_position",
    "time_to_next_event_seconds", "next_event_type", "next_event_id",
    "possession", "possession_team_id",  # StatsBomb possession ids carry look-ahead (experiment 3)
})


class FeatureLeakageError(ValueError):
    """A feature set declares a forbidden (target-derived or identifier) column."""


@dataclass(frozen=True)
class FeatureSet:
    name: str
    description: str
    input_columns: tuple[str, ...]
    status: str                                   # "implemented" | "planned"
    builder: Callable[[pd.DataFrame], tuple[np.ndarray, list[str]]] | None = None
    leakage_notes: str = ""

    def __post_init__(self) -> None:
        bad = set(self.input_columns) & FORBIDDEN_COLUMNS
        if bad:
            raise FeatureLeakageError(f"{self.name}: forbidden input columns {sorted(bad)}")
        if self.status not in ("implemented", "planned"):
            raise ValueError(f"{self.name}: unknown status {self.status!r}")
        if self.status == "implemented" and self.builder is None:
            raise ValueError(f"{self.name}: an implemented feature set needs a builder")


def one_hot(ids: np.ndarray, n_classes: int = K) -> np.ndarray:
    """(n, n_classes) float array with exactly one 1 per row; ids must lie in 0..n_classes-1."""
    ids = np.asarray(ids, dtype=np.int64)
    if ids.size and (ids.min() < 0 or ids.max() >= n_classes):
        raise ValueError(f"category id outside 0..{n_classes - 1}")
    out = np.zeros((len(ids), n_classes), dtype=np.float64)
    out[np.arange(len(ids)), ids] = 1.0
    return out


def _build_f0(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    ids = frame["symbol_id"].to_numpy()
    assert_valid_ids(ids)
    return one_hot(ids, K), [f"E_t={c}" for c in SCENARIO_D15_CLASSES]


F0 = FeatureSet(
    name="F0",
    description="current event only: x_t = OneHot(E_t), d = 15",
    input_columns=("symbol_id",),
    status="implemented",
    builder=_build_f0,
    leakage_notes="symbol_id is the class of the current event E_t; nothing else is read",
)

PLANNED = (
    FeatureSet("F1", "F0 + pitch third / pitch zone of E_t (from the raw event location)",
               ("symbol_id", "location_x", "location_y"), "planned",
               leakage_notes="raw event location of E_t only; missing location -> own category"),
    FeatureSet("F2", "F1 + continuous x, y, distance and angle to the attacked goal",
               ("symbol_id", "location_x", "location_y"), "planned",
               leakage_notes="StatsBomb event coordinates are already in the acting team's attacking "
                             "direction; standardise on the training fold only"),
    FeatureSet("F3", "F0 + period, match minute, time since the previous event",
               ("symbol_id", "period", "t_seconds"), "planned",
               leakage_notes="time since PREVIOUS event only; time to the next event is target-derived"),
    FeatureSet("F4", "F2 + 360 geometry of the current snapshot (opponent distances, local density)",
               ("symbol_id", "location_x", "location_y", "has_360"), "planned",
               leakage_notes="current freeze frame only; causal team identity and orientation; "
                             "visible-area masking; never a later snapshot"),
    FeatureSet("F5", "F0 + previous event classes E_{t-1}, E_{t-2}, ... (within the period)",
               ("symbol_id", "prev1_symbol_id", "prev2_symbol_id"), "planned",
               leakage_notes="history computed by shifting within the sequence; no future shift"),
    FeatureSet("F6", "match context: score state, team strength (optional future work)",
               ("symbol_id",), "planned",
               leakage_notes="score from EARLIER goals only; team strength from earlier matches only"),
)

REGISTRY: dict[str, FeatureSet] = {fs.name: fs for fs in (F0, *PLANNED)}


def design_matrix(frame: pd.DataFrame, feature_set: FeatureSet | str) -> tuple[np.ndarray, list[str]]:
    """Feature matrix X (n, d) and feature names for the rows of ``frame``.

    The builder receives a copy restricted to ``feature_set.input_columns``."""
    fs = REGISTRY[feature_set] if isinstance(feature_set, str) else feature_set
    if fs.status != "implemented" or fs.builder is None:
        raise NotImplementedError(f"feature set {fs.name} is planned, not implemented")
    missing = [c for c in fs.input_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"{fs.name}: columns {missing} not in the task table")
    x, names = fs.builder(frame.loc[:, list(fs.input_columns)].copy())
    if x.shape != (len(frame), len(names)):
        raise ValueError(f"{fs.name}: builder returned shape {x.shape} for {len(frame)} rows / {len(names)} names")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{fs.name}: non-finite feature values")
    return x, names
