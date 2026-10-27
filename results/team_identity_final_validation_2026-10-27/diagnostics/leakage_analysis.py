"""Leakage analysis from ml/rows.csv.gz (see ml_validation.py).

1. Exposure: rows whose OFFLINE identity features differ from the causal
   ones (only possible through records later than the row's event), and how
   often the event being predicted has a snapshot inside the 2 s window.
2. Orientation routes that can read later records.
3. Target information carried by the identity flags beyond the current
   event type: conditional mutual information I(flag; next_event_type |
   current_event_type) in bits, against a within-stratum permutation
   baseline (200 permutations).
Output: ../ml/leakage.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "ml"
df = pd.read_csv(OUT / "rows.csv.gz", low_memory=False)
n = len(df)
LATER = {"temporal_neighbor", "display_duplicate_snapshot", "display_duplicate_consensus", "display_camera_continuity", "user_frame_review"}
CAMERA = {"display_kickoff_own_half", "display_camera_side"}


def cmi(flag, target, stratum):
    total = 0.0
    for _, g in pd.DataFrame({"f": flag, "t": target, "s": stratum}).groupby("s"):
        p = len(g) / len(flag)
        joint = pd.crosstab(g.f, g.t, normalize=True).to_numpy()
        pf = joint.sum(1, keepdims=True)
        pt = joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = joint * np.log2(joint / (pf * pt))
        total += p * np.nansum(terms)
    return total


def permutation_baseline(flag, target, stratum, rounds=200, seed=0):
    rng = np.random.default_rng(seed)
    values = []
    f = flag.to_numpy().copy()
    groups = [np.flatnonzero(stratum.to_numpy() == s) for s in stratum.unique()]
    for _ in range(rounds):
        g = f.copy()
        for idx in groups:
            g[idx] = rng.permutation(f[idx])
        values.append(cmi(pd.Series(g), target, stratum))
    return float(np.mean(values)), float(np.percentile(values, 99))


feat = ["visible_teammates", "visible_opponents", "visible_ambiguous"]
offline_differs = np.zeros(n, bool)
for c in feat:
    offline_differs |= (df[c] != df[c + "_offline"]).to_numpy()
legacy_changed = (df.visible_teammates_offline != df.legacy_visible_teammates) | (df.visible_opponents_offline != df.legacy_visible_opponents)
causal_vs_legacy = (df.visible_teammates != df.legacy_visible_teammates) | (df.visible_opponents != df.legacy_visible_opponents)
S = {
    "rows": n,
    "identity_exposure": {
        "rows_offline_features_differ_from_causal": int(offline_differs.sum()),
        "pct": round(100 * offline_differs.mean(), 4),
        "of_which_next_event_snapshot_within_2s": int((offline_differs & df.next_event_snapshot_in_window).sum()),
        "rows_next_event_snapshot_within_2s_overall": int(df.next_event_snapshot_in_window.sum()),
        "pct_rows_next_event_snapshot_within_2s": round(100 * df.next_event_snapshot_in_window.mean(), 2),
        "rows_offline_uncertain": int((df.visible_ambiguous_offline > 0).sum()),
        "rows_causal_uncertain": int((df.visible_ambiguous > 0).sum()),
        "rows_counts_changed_vs_legacy_offline": int(legacy_changed.sum()),
        "rows_counts_changed_vs_legacy_causal": int(causal_vs_legacy.sum()),
        "causal_vs_offline_quality_crosstab": pd.crosstab(df.identity_quality, df.identity_quality_offline).to_dict(),
    },
    "orientation_exposure": {
        "rows_by_route": df.orientation_route.value_counts().to_dict(),
        "rows_route_can_read_later_records": int(df.orientation_route.isin(LATER).sum()),
        "rows_route_uses_period_camera_side": int(df.orientation_route.isin(CAMERA).sum()),
        "rows_actor_xy_differs_from_legacy": int((~np.isclose(df.actor_x.fillna(-1), df.legacy_actor_x.fillna(-1))).sum()),
        "rows_actor_xy_differs_and_route_later": int(((~np.isclose(df.actor_x.fillna(-1), df.legacy_actor_x.fillna(-1))) & df.orientation_route.isin(LATER)).sum()),
    },
}
target, stratum = df.next_event_type, df.current_event_type
flags = {
    "offline_uncertain": (df.visible_ambiguous_offline > 0).astype(int),
    "causal_uncertain": (df.visible_ambiguous > 0).astype(int),
    "offline_differs_from_causal": pd.Series(offline_differs.astype(int)),
    "offline_counts_changed_vs_legacy": legacy_changed.astype(int),
    "causal_counts_changed_vs_legacy": causal_vs_legacy.astype(int),
}
S["conditional_mutual_information_bits"] = {}
for name, flag in flags.items():
    value = cmi(flag, target, stratum)
    mean, p99 = permutation_baseline(flag, target, stratum)
    S["conditional_mutual_information_bits"][name] = {
        "cmi": value, "permutation_mean": mean, "permutation_p99": p99, "above_p99": value > p99,
        "flagged_rows": int(flag.sum()),
    }
# reference scale: information in a genuinely predictive feature
S["conditional_mutual_information_bits"]["reference_actor_x_thirds"] = {
    "cmi": cmi(pd.cut(df.actor_x, [-1, 40, 80, 121]).astype(str), target, stratum)}
(OUT / "leakage.json").write_text(json.dumps(S, indent=1, default=str), encoding="utf-8")
print(json.dumps(S, indent=1, default=str))
