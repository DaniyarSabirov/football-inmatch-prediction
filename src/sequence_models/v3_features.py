r"""Experiment 3: causal spatial, possession, temporal and history features.

Every feature of event *t* uses only event *t* itself and EARLIER raw records of the same period
(forward scans and shifts; tested by truncation in tests/test_v3_features.py).

**Coordinates.** The raw event ``location`` is in the acting team's attacking frame (StatsBomb
spec; experiment 1: 99.98% agreement with the causal 360 actor point; experiment 2: 99.97%
zone agreement). ``orientation_audit`` re-checks the direction with independent football facts
(shots, goalkeeper actions, kick-offs, clearances). Freeze-frame (360) positions are not used here.

**Pitch geometry** (StatsBomb 120 x 80 units, nominally yards; 1 m = 1/0.9144 units):
opponent goal centre (120, 40), posts (120, 36) and (120, 44) (goal width 8 units = 7.32 m),
penalty area x >= 102, 18 <= y <= 62. Extended "danger" area: penalty area grown by 5 m =
5.468 units on the three open sides, i.e. x >= 96.53 and 12.53 <= y <= 67.47.

- distance to goal: :math:`\sqrt{(120-x)^2 + (40-y)^2}`
- shooting angle: angle at the event location between the two posts,
  :math:`|\operatorname{atan2}(44-y, 120-x) - \operatorname{atan2}(36-y, 120-x)|` in radians.

**Causal possession** (StatsBomb ``possession`` ids are assigned by annotators with the whole match
available, see ``possession_audit``): the controlling team after event t is the team of the
most recent ball-control record up to t. A new causal possession starts when that team changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOAL_X, GOAL_Y, POST_LOW, POST_HIGH = 120.0, 40.0, 36.0, 44.0
YARDS_PER_METRE = 1 / 0.9144
DANGER_MARGIN = 5 * YARDS_PER_METRE                      # 5.468 units
BOX_X, BOX_Y_LOW, BOX_Y_HIGH = 102.0, 18.0, 62.0
DANGER_X, DANGER_Y_LOW, DANGER_Y_HIGH = BOX_X - DANGER_MARGIN, BOX_Y_LOW - DANGER_MARGIN, BOX_Y_HIGH + DANGER_MARGIN
CONTROL_TYPES = {"Pass", "Ball Receipt*", "Carry", "Dribble", "Shot", "Ball Recovery", "Miscontrol", "Clearance",
                 "Shield"}
GK_ACTIONS = {"Collected", "Keeper Sweeper", "Punch", "Smother", "Shot Saved", "Save", "Penalty Saved",
              "Shot Saved Off Target", "Shot Saved to Post", "Penalty Saved to Post"}
INTERCEPTION_WON = {"Won", "Success In Play", "Success Out"}
DIST_EDGES = (12.0, 18.0, 25.0, 35.0, 50.0, 70.0)        # 7 distance-to-goal bins
UNKNOWN = -1


# --------------------------------------------------------------------------- geometry
def distance_to_goal(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(GOAL_X - x, GOAL_Y - y)


def shooting_angle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = GOAL_X - x
    return np.abs(np.arctan2(POST_HIGH - y, dx) - np.arctan2(POST_LOW - y, dx))


def in_danger_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x >= DANGER_X) & (y >= DANGER_Y_LOW) & (y <= DANGER_Y_HIGH)


def in_penalty_area(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (x >= BOX_X) & (y >= BOX_Y_LOW) & (y <= BOX_Y_HIGH)


def _bin(values: np.ndarray, edges, unknown_if_nan: bool = True) -> np.ndarray:
    out = np.searchsorted(np.asarray(edges, dtype=float), values, side="right").astype(np.int64)
    if unknown_if_nan:
        out[np.isnan(values)] = UNKNOWN
    return out


def spatial_features(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Continuous geometry and the candidate discrete spatial representations (UNKNOWN = no location)."""
    x = np.clip(np.asarray(x, dtype=float), 0, 120)
    y = np.clip(np.asarray(y, dtype=float), 0, 80)
    known = ~np.isnan(x) & ~np.isnan(y)
    lateral = np.abs(y - 40)
    dist = distance_to_goal(x, y)
    danger = in_danger_area(x, y)
    z3 = _bin(x, (40, 80))
    b6 = _bin(x, (20, 40, 60, 80, 100))
    central = (lateral <= 22).astype(np.int64)                       # within the penalty-area width
    g12 = np.where(known, b6 * 2 + central, UNKNOWN)
    d7 = _bin(dist, DIST_EDGES)
    z3d = np.where(known, np.where(danger, 3, z3), UNKNOWN)          # thirds with the danger area split off
    return {"x": x, "lateral": lateral, "distance_to_goal": dist, "shooting_angle": shooting_angle(x, y),
            "danger_area": danger & known, "penalty_area": in_penalty_area(x, y) & known,
            "z3": np.where(known, z3, UNKNOWN), "b6": np.where(known, b6, UNKNOWN), "g12": g12,
            "d7": np.where(known, d7, UNKNOWN), "z3d": z3d}


SPATIAL_CARDINALITY = {"z3": 3, "b6": 6, "g12": 12, "d7": 7, "z3d": 4}


# --------------------------------------------------------------------------- causal possession and time
def is_control(t: pd.DataFrame) -> np.ndarray:
    """Records by which a team demonstrably plays the ball (known at the record itself)."""
    typ, sub, out = t["type"], t["subtype"], t["outcome"]
    control = typ.isin(CONTROL_TYPES) & ~((typ == "Ball Receipt*") & (out == "Incomplete"))
    control |= (typ == "Goal Keeper") & sub.isin(GK_ACTIONS)
    control |= (typ == "Interception") & out.isin(INTERCEPTION_WON)
    return control.to_numpy()


def causal_raw_features(t: pd.DataFrame) -> pd.DataFrame:
    """Per raw event (table in raw order, columns incl. type/subtype/outcome/team_id/t_seconds/location_x):
    causal possession and temporal features. Periods are independent."""
    out = []
    control = is_control(t)
    for (_, _), idx in t.groupby(["match_id", "period"], sort=False).indices.items():
        g = t.iloc[idx]
        team = g["team_id"].to_numpy()
        ctrl = control[idx]
        ts = g["t_seconds"].to_numpy(dtype=float)
        x = g["location_x"].to_numpy(dtype=float)
        n = len(g)
        poss_team = np.empty(n, dtype=float)
        poss_id = np.empty(n, dtype=np.int64)
        start_pos = np.empty(n, dtype=np.int64)
        current, pid, start = np.nan, 0, 0
        for i in range(n):
            if ctrl[i] and team[i] != current:
                current, pid, start = team[i], pid + 1, i
            poss_team[i], poss_id[i], start_pos[i] = current, pid, start
        # timestamps: corrupted (backwards) stamps become unknown rather than negative durations
        running_max = np.maximum.accumulate(np.where(np.isnan(ts), -np.inf, ts))
        bad = ts < running_max - 1e-9
        tsc = np.where(bad, np.nan, ts)
        duration = tsc - tsc[start_pos]
        duration[(duration < 0) | np.isnan(duration)] = np.nan
        # location in the frame of the team in possession (opponent records are mirrored)
        x_poss = np.where(team == poss_team, x, 120.0 - x)
        x_ffill = pd.Series(x_poss).ffill().to_numpy()
        start_x = np.full(n, np.nan)
        for s in np.unique(start_pos):
            seg = np.flatnonzero(start_pos == s)
            first = seg[~np.isnan(x_poss[seg])]
            if len(first):
                fx = x_poss[first[0]]
                start_x[seg[seg >= first[0]]] = fx
        prev_team = np.concatenate([[np.nan], team[:-1].astype(float)])
        changed = np.concatenate([[True], poss_id[1:] != poss_id[:-1]])
        recent = pd.Series(changed).rolling(3, min_periods=1).max().astype(bool).to_numpy()
        dt = tsc - np.concatenate([[np.nan], tsc[:-1]])
        dt[dt < 0] = np.nan
        valid_t = np.where(np.isnan(tsc), running_max, tsc)
        rate = np.arange(1, n + 1) - np.searchsorted(valid_t, valid_t - 10.0, side="right")
        out.append(pd.DataFrame({
            "event_id": g["event_id"].to_numpy(), "causal_possession_team": poss_team, "causal_possession_id": poss_id,
            "possession_duration_s": duration, "possession_events_so_far": np.arange(n) - start_pos + 1,
            "possession_progression": x_ffill - start_x, "x_possession_frame": x_ffill,
            "same_team_as_previous": team == prev_team, "recent_possession_change": recent,
            "time_since_previous_s": dt, "events_last_10s": rate.astype(float),
            "timestamp_corrupted": bad,
        }))
    return pd.concat(out, ignore_index=True)


def possession_bins(f: pd.DataFrame) -> dict[str, np.ndarray]:
    dur = f["possession_duration_s"].to_numpy(dtype=float)
    prog = f["possession_progression"].to_numpy(dtype=float)
    n_ev = f["possession_events_so_far"].to_numpy(dtype=float)
    dur4 = _bin(dur, (5, 15, 30))
    prog4 = _bin(prog, (-5, 10, 30))
    return {"poss_dur4": dur4, "poss_n4": _bin(n_ev, (2, 4, 8)), "poss_prog4": prog4,
            "poss_dur_prog16": np.where((dur4 >= 0) & (prog4 >= 0), dur4 * 4 + prog4, UNKNOWN),
            "same_team2": f["same_team_as_previous"].to_numpy().astype(np.int64),
            "recent_change2": f["recent_possession_change"].to_numpy().astype(np.int64),
            "dt4": _bin(f["time_since_previous_s"].to_numpy(dtype=float), (1e-9, 1.0, 3.0)),
            "rate3": _bin(f["events_last_10s"].to_numpy(dtype=float), (6, 11)),
            "minute7": minute_bins(f)}


def minute_bins(f: pd.DataFrame) -> np.ndarray:
    m, p = f["minute"].to_numpy(dtype=float), f["period"].to_numpy()
    b = np.where(p == 1, np.minimum(m // 15, 2), np.where(p == 2, 3 + np.minimum((m - 45) // 15, 2), 6))
    return np.clip(b, 0, 6).astype(np.int64)


CONTEXT_CARDINALITY = {"pf_d7": 7, "pf_z3d": 4, "poss_dur4": 4, "poss_n4": 4, "poss_prog4": 4, "poss_dur_prog16": 16, "same_team2": 2,
                       "recent_change2": 2, "dt4": 4, "rate3": 3, "minute7": 7, **SPATIAL_CARDINALITY}


def history_tokens(symbol: np.ndarray, seq: np.ndarray, poss: np.ndarray | None, n_symbols: int, depth: int = 3,
                   reset_at_possession: bool = False) -> dict[str, np.ndarray]:
    """prev1..prev<depth>: symbols t-1..t-depth of the same sequence; START (= V) before the sequence
    start; with reset, NEWPOSS (= V + 1) for every event before the current causal possession."""
    out = {}
    n = len(symbol)
    start_tok, poss_tok = n_symbols, n_symbols + 1
    for k in range(1, depth + 1):
        tok = np.full(n, start_tok, dtype=np.int64)
        tok[k:] = symbol[:-k]
        same = np.zeros(n, dtype=bool)
        same[k:] = seq[k:] == seq[:-k]
        tok[~same] = start_tok
        if reset_at_possession and poss is not None:
            same_poss = np.zeros(n, dtype=bool)
            same_poss[k:] = poss[k:] == poss[:-k]
            tok[same & ~same_poss] = poss_tok
            if k > 1:  # once the history left the possession, older events are also outside it
                tok[(out[f"prev{k - 1}"] == poss_tok)] = poss_tok
        out[f"prev{k}"] = tok
    return out


# --------------------------------------------------------------------------- audits
def orientation_audit(t: pd.DataFrame) -> dict:
    """Independent football checks that event x points towards the acting team's opponent goal."""
    p = t[t["period"] < 5]
    shots = p[p["type"] == "Shot"]
    gk = p[(p["type"] == "Goal Keeper") & p["subtype"].isin(GK_ACTIONS)]
    ko = p[(p["type"] == "Pass") & (p["subtype"] == "Kick Off")]
    clr = p[p["type"] == "Clearance"]
    by_period = shots.groupby("period")["location_x"].apply(lambda s: float((s >= 60).mean())).to_dict()
    return {
        "shots": int(len(shots)), "shots_x_ge_60_share": float((shots["location_x"] >= 60).mean()),
        "shots_x_ge_60_share_by_period": {str(k): v for k, v in by_period.items()},
        "shots_x_lt_60_cases": shots.loc[shots["location_x"] < 60, ["match_id", "period", "index", "location_x", "subtype"]]
        .to_dict("records"),
        "goalkeeper_actions": int(len(gk)), "goalkeeper_actions_x_le_30_share": float((gk["location_x"] <= 30).mean()),
        "kick_offs": int(len(ko)), "kick_offs_x_in_55_65_share": float(ko["location_x"].between(55, 65).mean()),
        "clearances": int(len(clr)), "clearances_x_le_50_share": float((clr["location_x"] <= 50).mean()),
        "missing_location_share": float(p["location_x"].isna().mean()),
    }


def possession_audit(t: pd.DataFrame, f: pd.DataFrame) -> dict:
    """Compare StatsBomb possession ids with the causal reconstruction.

    Look-ahead signature: a StatsBomb possession credited to team X whose FIRST record is not a
    ball-control record of X (e.g. it starts with the opponent's Pressure or with a restart
    annotation). Crediting X at that record requires knowing that X will gain control later.
    """
    m = t[["event_id", "match_id", "period", "type", "subtype", "outcome", "team_id", "possession",
           "possession_team_id"]].merge(f, on="event_id")
    m = m[m["period"] < 5].reset_index(drop=True)
    control = is_control(m)
    own_control = control & (m["team_id"] == m["possession_team_id"]).to_numpy()
    known = m["causal_possession_team"].notna().to_numpy()
    agree = float((m["possession_team_id"].to_numpy()[known] == m["causal_possession_team"].to_numpy()[known]).mean())
    g = m.assign(own_control=own_control).groupby(["match_id", "possession"], sort=False)
    first_is_own_control = g["own_control"].first()
    first_type = g["type"].first()
    sb_change = (m.groupby(["match_id", "period"])["possession"].diff() != 0).to_numpy()
    causal_change = (m.groupby(["match_id", "period"])["causal_possession_id"].diff() != 0).to_numpy()
    return {
        "events_compared": int(known.sum()),
        "statsbomb_possession_team_equals_causal_controlling_team": agree,
        "statsbomb_possessions": int(len(first_is_own_control)),
        "statsbomb_possessions_first_record_not_own_control": int((~first_is_own_control).sum()),
        "share_first_record_not_own_control": float((~first_is_own_control).mean()),
        "first_record_types_when_not_own_control": first_type[~first_is_own_control].value_counts().head(8).to_dict(),
        "statsbomb_boundaries": int(sb_change.sum()), "causal_boundaries": int(causal_change.sum()),
        "statsbomb_boundaries_also_causal_boundary_same_record": float((sb_change & causal_change).sum() / sb_change.sum()),
        "verdict": "StatsBomb possession ids are not used as features: they are annotator-assigned with the whole "
                   "match available; possession features are rebuilt causally from earlier records",
    }
