r"""Causal canonical-event history features for experiment 5 (XGB-F0, XGB-H5).

Notation (``docs/thesis/02_mathematical_framework.md``, ``docs/canonical_prediction_protocol.md``):
:math:`t` is the position of an observation in the canonical sequence of one (match, period)
(column ``seq_pos``, 0-based). It is neither elapsed time nor the raw StatsBomb ``index``:
``index`` orders the RAW records of a match, and many raw records (Ball Receipt*, companions
K1-K3, administrative records) are not observations, so consecutive observations usually have
non-consecutive ``index`` values.

For an eligible row at position :math:`t` the feature configurations are

- ``XGB-F0``: :math:`\mathbf{x}_t = (E_t)`;
- ``XGB-H5``: :math:`\mathbf{x}_t = (E_{t-4}, E_{t-3}, E_{t-2}, E_{t-1}, E_t)`,

with target :math:`y_t = E_{t+1}` (``target_id``). History positions are previous observations
of the SAME canonical sequence, whatever their 360 status (an earlier event is available at
prediction time; this is the same rule as the canonical Markov-2 context). Where
:math:`t-j < 0` the position holds the padding token ``PAD = 15`` ("<START>"), which is not
one of the 15 class ids, so history never comes from an earlier period or match and early
rows are kept, not dropped.

Encoding (fixed vocabulary, nothing is fitted, identical in every fold): the :math:`E_t` block
is one-hot over the 15 classes; each history block is one-hot over 16 categories
(15 classes + <START>). XGB-F0 has 15 columns, XGB-H5 has 15 + 4 * 16 = 79.

Only ``sequence_id``, ``seq_pos`` and ``symbol_id`` are read to build features; ``target_id``
is read only as the label. No identifier is a feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .canonical_task import CLASSES, K, NO_TARGET

PAD = K                       # 15
PAD_NAME = "<START>"
N_CATEGORIES = K + 1          # 16 symbols per history position

FEATURE_SETS: dict[str, int] = {"XGB-F0": 0, "XGB-H5": 4}   # name -> number of previous positions


class HistoryError(ValueError):
    """The table is not a valid canonical sequence table for history construction."""


def position_label(lag: int) -> str:
    return "E_t" if lag == 0 else f"E_{{t-{lag}}}"


def history_ids(table: pd.DataFrame, n_previous: int) -> np.ndarray:
    """(n_observations, n_previous + 1) int array: columns E_{t-n_previous}, ..., E_{t-1}, E_t.

    ``table``: canonical observations (all of them, not only eligible rows), grouped by
    sequence with ``seq_pos`` = 0, 1, 2, ... in table order. Positions before the sequence
    start hold ``PAD``."""
    seq = table["sequence_id"].to_numpy()
    pos = table["seq_pos"].to_numpy()
    sym = table["symbol_id"].to_numpy().astype(np.int64)
    if len(sym) and (sym.min() < 0 or sym.max() >= K):
        raise HistoryError("symbol id outside 0..14")
    start = np.r_[True, seq[1:] != seq[:-1]]
    if len(pd.unique(seq[start])) != int(start.sum()):
        raise HistoryError("a sequence is not contiguous in the table")
    expected = np.arange(len(seq)) - np.maximum.accumulate(np.where(start, np.arange(len(seq)), 0))
    if not np.array_equal(pos, expected):
        raise HistoryError("seq_pos is not 0, 1, 2, ... within each sequence in table order")
    out = np.full((len(sym), n_previous + 1), PAD, dtype=np.int64)
    for lag in range(n_previous + 1):
        col = n_previous - lag
        if lag == 0:
            out[:, col] = sym
        else:
            ok = pos >= lag
            out[ok, col] = sym[np.flatnonzero(ok) - lag]   # same sequence because pos >= lag
    return out


def encode(ids: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """One-hot design matrix (float32) and column names for history ids from ``history_ids``."""
    ids = np.asarray(ids, dtype=np.int64)
    n_previous = ids.shape[1] - 1
    if (ids[:, -1] == PAD).any():
        raise HistoryError("the current event E_t can never be padding")
    blocks, names = [], []
    for col in range(n_previous + 1):
        lag = n_previous - col
        width = K if lag == 0 else N_CATEGORIES
        if (ids[:, col] < 0).any() or (ids[:, col] >= width).any():
            raise HistoryError(f"category outside the vocabulary at {position_label(lag)}")
        block = np.zeros((len(ids), width), dtype=np.float32)
        block[np.arange(len(ids)), ids[:, col]] = 1.0
        blocks.append(block)
        labels = list(CLASSES) + ([PAD_NAME] if lag else [])
        names += [f"{position_label(lag)}={c}" for c in labels]
    return np.hstack(blocks), names


def feature_position(names: list[str]) -> list[str]:
    """Position label (e.g. 'E_{t-2}') of every feature column."""
    return [n.split("=", 1)[0] for n in names]


@dataclass
class HistoryRows:
    """Eligible canonical rows with their 5-position history (the widest configuration).

    ``ctx[:, 4]`` is E_t, ``ctx[:, 3]`` is E_{t-1}, ..., ``ctx[:, 0]`` is E_{t-4}. XGB-F0 uses
    ``ctx[:, 4:]`` and XGB-H5 ``ctx`` - the row set is identical by construction."""

    event_id: np.ndarray
    match_id: np.ndarray
    sequence_id: np.ndarray
    seq_pos: np.ndarray
    ctx: np.ndarray
    y: np.ndarray

    @classmethod
    def from_table(cls, table: pd.DataFrame, n_previous: int = 4) -> "HistoryRows":
        ids = history_ids(table, n_previous)
        e = table["row_eligible"].to_numpy().astype(bool)
        y = table["target_id"].to_numpy()[e].astype(np.int64)
        if (y == NO_TARGET).any() or (y < 0).any() or (y >= K).any():
            raise HistoryError("eligible row without a valid target")
        return cls(table["event_id"].to_numpy()[e], table["match_id"].to_numpy()[e],
                   table["sequence_id"].to_numpy()[e], table["seq_pos"].to_numpy()[e], ids[e], y)

    def features(self, feature_set: str) -> np.ndarray:
        n_previous = FEATURE_SETS[feature_set]
        return self.ctx[:, self.ctx.shape[1] - 1 - n_previous:]

    def mask(self, match_ids) -> np.ndarray:
        return np.isin(self.match_id, list(match_ids))


def aggregate_cells(ids: np.ndarray, y: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Unique (features[, target]) cells.

    Returns (cell feature ids, cell targets or None, cell counts, inverse) where
    ``inverse[i]`` is the cell of row ``i``. Cells are sorted by code, so the result does not
    depend on row order."""
    cols = ids if y is None else np.column_stack([ids, y])
    code = np.zeros(len(cols), dtype=np.int64)
    for j in range(cols.shape[1]):
        code = code * N_CATEGORIES + cols[:, j]
    uniq, first, inverse, counts = np.unique(code, return_index=True, return_inverse=True, return_counts=True)
    cells = cols[first]
    if y is None:
        return cells, None, counts.astype(np.float64), inverse
    return cells[:, :-1], cells[:, -1], counts.astype(np.float64), inverse
