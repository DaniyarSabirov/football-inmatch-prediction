"""Classify every residual colour transition (transitions.csv.gz) by the
evidence available, and decompose the total into an estimated genuine-label
part and a pairing part. Output: ../transition_classes.csv, ../transition_summary.json

Classes (first match wins):
  confirmed_incorrect   an independent anchor shows one side is wrong: a
                        goalkeeper whose colour contradicts the goal he stands
                        at (own-frame goal side; not the identity rule's
                        decision, which leaves such keepers unresolved), or
                        an actor point whose colour is not its own event's
                        team in an unlinked record
  strongly_suspected    pairing near-certain (dt <= 0.25 s and margin > 2 yd)
                        or two-sided in the sequence (blip): one record is
                        wrong about this player, which one is unknown
  ambiguous             dt <= 1 s and margin > 3 yd, no anchor
  likely_pairing_artefact
                        dt > 1 s or margin <= 3 yd: strata where the
                        transition rate is 10-400x the near-certain floor

Decomposition: the floor rate p0 = transition rate among pairs with dt <= 0.25 s
and margin > 5 yd (pairing errors there are negligible). If genuine label
disagreements occurred at p0 in every stratum, the genuine part is
p0 * all pairs; the remainder is attributed to pairing. This is an ESTIMATE
under that assumption, not a count.
"""
import json
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1]
t = pd.read_csv(OUT / "transitions.csv.gz", low_memory=False)
s = pd.read_csv(OUT / "pair_strata.csv")

keeper_anchor = ((t.goal_a.notna() & (t.goal_a != t.colour_a)) | (t.goal_b.notna() & (t.goal_b != t.colour_b)))
actor_anchor = ((t.actor_a & ~t.linked_a & (t.colour_a != t.team_a)) | (t.actor_b & ~t.linked_b & (t.colour_b != t.team_b)))
near_certain = (t.dt_s <= 0.25) & (t.pair_margin > 2)
blip = t.pattern.isin(["blip_A", "blip_B"])
medium = (t.dt_s <= 1.0) & (t.pair_margin > 3)
t["class"] = "likely_pairing_artefact"
t.loc[medium, "class"] = "ambiguous"
t.loc[near_certain | blip, "class"] = "strongly_suspected"
t.loc[keeper_anchor | actor_anchor, "class"] = "confirmed_incorrect"
t.to_csv(OUT / "transition_classes.csv", index=False)

pairs = s.pairs.sum()
floor = s[(s.dt_s.isin(["<=0.0", "<=0.25"])) & (s.pair_margin == ">5.0")]
p0 = floor[floor.transition].pairs.sum() / floor.pairs.sum()
summary = {
    "pairs_evaluated": int(pairs), "transitions": len(t), "transition_rate_pct": round(100 * len(t) / pairs, 4),
    "matches_with_transitions": int(t.match_id.nunique()),
    "snapshot_links_with_transitions": int(t[["event_a", "event_b"]].drop_duplicates().shape[0]),
    "snapshots_involved": int(pd.unique(pd.concat([t.event_a, t.event_b])).size),
    "player_observations_involved": int(pd.unique(pd.concat([t.event_a + ":" + t.point_a.astype(str), t.event_b + ":" + t.point_b.astype(str)])).size),
    "classes": t["class"].value_counts().to_dict(),
    "confirmed_by_anchor": {"keeper_goal_side": int(keeper_anchor.sum()), "actor_semantics": int(actor_anchor.sum())},
    "patterns": t.pattern.value_counts().to_dict(),
    "floor_rate_pct (dt<=0.25 s, margin>5 yd)": round(100 * p0, 4),
    "floor_pairs": int(floor.pairs.sum()), "floor_transitions": int(floor[floor.transition].pairs.sum()),
    "estimated_genuine_label_disagreements": round(p0 * pairs),
    "estimated_pairing_artefacts": round(len(t) - p0 * pairs),
    "share_attributed_to_pairing_pct": round(100 * (1 - p0 * pairs / len(t)), 1),
    "per_match_transition_rate_pct": None,
}
m = pd.read_csv(OUT / "match_transitions.csv")
per_pair = s.pairs.sum() / len(m)
summary["transitions_per_match"] = m.transitions.describe().round(1).to_dict()
summary["top10_matches_share_pct"] = round(100 * m.transitions.nlargest(10).sum() / m.transitions.sum(), 2)
summary["by_event_type_a_top"] = t.type_a.value_counts().head(8).to_dict()
(OUT / "transition_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print(json.dumps(summary, indent=1))
print(pd.crosstab(t["class"], t.pattern))
