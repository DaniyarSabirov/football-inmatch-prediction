"""Longitudinal pitch thirds relative to the acting team's attacking direction.

Source coordinate: the raw event ``location`` x. The StatsBomb event
specification records every event's location in the acting team's attacking
frame (the team attacks toward x = 120); the first experiment validated this
against the causal 360 actor point (99.98% within 5 units). For a defensive
action (Pressure, Duel, Block, ...) the acting team is the defending team, so
the zone is relative to *that* team's attacking direction.

The location belongs to the current event itself, so it is available at the
prediction time (right after the event). No orientation inference, no 360
snapshot and no later record is involved.

Zones (pitch units, length 120):

    DEFENSIVE_THIRD   0 <= x < 40
    MIDDLE_THIRD     40 <= x < 80
    ATTACKING_THIRD  80 <= x <= 120

Values within OUT_OF_PITCH_TOLERANCE (1 unit) beyond an end line belong to
the adjacent third: the development audit found 2 shots at x = 120.1 and
120.5. Anything further out raises. A missing location is ZONE_UNKNOWN (-1),
never imputed.

These are spatial zones, not tactical phases.
"""

from __future__ import annotations

import numpy as np

ZONE_NAMES = ("Defensive third", "Middle third", "Attacking third")
ZONE_UNKNOWN = -1
PITCH_LENGTH = 120.0
BOUNDARIES = (40.0, 80.0)
OUT_OF_PITCH_TOLERANCE = 1.0


class ZoneError(ValueError):
    """A coordinate lies outside the pitch."""


def zone_of_x(x: np.ndarray) -> np.ndarray:
    """Zone index (0, 1, 2) per x; ZONE_UNKNOWN for NaN."""
    x = np.asarray(x, dtype=float)
    known = ~np.isnan(x)
    tol = OUT_OF_PITCH_TOLERANCE
    if np.any((x[known] < -tol) | (x[known] > PITCH_LENGTH + tol)):
        raise ZoneError(f"x outside [-{tol}, 120 + {tol}]")
    zones = np.full(x.shape, ZONE_UNKNOWN, dtype=np.int64)
    zones[known] = np.searchsorted(np.array(BOUNDARIES), x[known], side="right")
    return zones
