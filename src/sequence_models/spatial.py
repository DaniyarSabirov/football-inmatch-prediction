"""Causal spatial observations for the spatial HMM.

Source: the production dataset's approved causal columns ``actor_x`` and
``actor_y`` (causal orientation; empty when the orientation of the current
snapshot cannot be decided from earlier records). They are expressed on the
match's *reference team* scale (src/team_normalization.py), so the x axis
points toward the reference team's attacking goal for both teams' events.
To make "position on the pitch" mean the same thing for every acting team,
the point is mapped back into the acting team's attacking frame:

    (x_att, y_att) = (actor_x, actor_y)              if team_normalization == REFERENCE
                   = (120 - actor_x, 80 - actor_y)   if team_normalization == MIRRORED

``team_normalization`` (classified METADATA) is used **only** for this
deterministic transform, never as a model input. The reference team is the
team of the match's first 360 event, which precedes every row, so the
transform is causal. The data audit validates the transform against the raw
event ``location`` (which the StatsBomb spec defines in the acting team's
attacking frame). The derived features are pending allowlist approval.

Features (StatsBomb pitch units, 120 x 80):

- ``x_attacking``: 0 = own goal line, 120 = opponent goal line;
- ``lateral_distance_from_centre``: |y_att - 40|, 0 = centre line, 40 = touchline.

Unobserved means missing (NaN), never zero: events without a 360 record, with
an undecided orientation or without a single unambiguous actor point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml_features import approved_feature_columns, load_feature_classification, select_feature_matrix
from src.utils.paths import PROCESSED_DATA_DIR

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
SPATIAL_FEATURES = ("x_attacking", "lateral_distance_from_centre")
TRANSFORM_COLUMN = "team_normalization"
REFERENCE, MIRRORED = "REFERENCE", "MIRRORED"


def production_match_path(match_id: int) -> str:
    return str(PROCESSED_DATA_DIR / f"statsbomb_360_match_{match_id}_next_events.csv")


def load_production_rows(match_ids: list[int]) -> pd.DataFrame:
    """Per-match production CSVs for exactly these matches (no other match is read)."""
    frames = [pd.read_csv(production_match_path(m), low_memory=False) for m in match_ids]
    return pd.concat(frames, ignore_index=True)


def spatial_observations(production_rows: pd.DataFrame) -> pd.DataFrame:
    """event_id + causal spatial features from approved columns (allowlist-validated)."""
    classification = load_feature_classification()
    approved = select_feature_matrix(production_rows, classification)  # validates schema + allowlist
    for column in ("actor_x", "actor_y", "orientation_available"):
        if column not in approved_feature_columns(classification):
            raise ValueError(f"{column} is no longer an approved feature; spatial features are invalid")
    normalization = production_rows[TRANSFORM_COLUMN]
    unknown = set(normalization.unique()) - {REFERENCE, MIRRORED}
    if unknown:
        raise ValueError(f"unexpected team_normalization labels {unknown}")
    mirrored = (normalization == MIRRORED).to_numpy()
    actor_x = approved["actor_x"].to_numpy(dtype=float)
    actor_y = approved["actor_y"].to_numpy(dtype=float)
    available = approved["orientation_available"].astype(bool).to_numpy()
    x_att = np.where(mirrored, PITCH_LENGTH - actor_x, actor_x)
    y_att = np.where(mirrored, PITCH_WIDTH - actor_y, actor_y)
    x_att[~available] = np.nan
    y_att[~available] = np.nan
    return pd.DataFrame({
        "event_id": production_rows["current_event_id"].to_numpy(),
        "x_attacking": x_att,
        "lateral_distance_from_centre": np.abs(y_att - PITCH_WIDTH / 2),
    })


@dataclass(frozen=True)
class SpatialScaler:
    """Standardisation fitted on training observations only (NaN ignored)."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "SpatialScaler":
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("expected a non-empty (n, d) array")
        mean = np.nanmean(values, axis=0)
        std = np.nanstd(values, axis=0)
        if not np.all(np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("cannot fit scaler: a feature is constant or entirely missing in training data")
        return cls(mean=mean, std=std)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}
