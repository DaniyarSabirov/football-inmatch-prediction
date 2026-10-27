"""Independent check of team identities at kick-offs (Law 8: at a kick-off
every player is in his own half; the kicker stands on the centre spot).

In the record's own frame the kicking team defends x = 0, so its players
must have x < 60 and the opponents x > 60. Players within 2 yd of the
halfway line and the actor are not evaluated. Kick-offs whose orientation
was itself decided from flags + Law 8 (route display_kickoff_own_half) or by
a reviewer (user_frame_review) are excluded as circular; only kick-offs whose
orientation came from geometry or another flag-free route count.

Output: ../kickoff_law8.csv (one row per evaluated kick-off), printed summary.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.orientation_correction import apply_correction  # noqa: E402
from src.orientation_resolution import match_team_ids, resolve_match_orientation  # noqa: E402
from src.team_identity import IDENTITY_UNRESOLVED, resolve_match_team_identity  # noqa: E402

DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
EXCLUDED_ROUTES = {"display_kickoff_own_half", "user_frame_review"}


def audit(match_id):
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    mo = resolve_match_orientation(match_id, events, frames)
    recs = mo.records.usable_frames
    names = {r.event["team"]["id"]: r.event["team"]["name"] for r in recs}
    pair = tuple(names[t] for t in match_team_ids(mo.records))
    inputs = ([(r.event, r.frame_360) for r in recs], [x.orientation for x in mo.resolved], mo.problem_a, pair, events)
    offline = resolve_match_team_identity(*inputs)
    causal = resolve_match_team_identity(*inputs, causal=True)
    rows = []
    for record, resolved in zip(recs, mo.resolved):
        event = record.event
        if ((event.get("pass") or {}).get("type") or {}).get("name") != "Kick Off" or resolved.orientation == "UNCERTAIN":
            continue
        own = apply_correction(record.frame_360, resolved.orientation)["freeze_frame"]
        team = event["team"]["name"]
        counts = {}
        for label, identity in (("legacy", offline[event["id"]].legacy_teams), ("spec", offline[event["id"]].spec_teams),
                                ("offline", offline[event["id"]].teams), ("causal", causal[event["id"]].teams)):
            evaluated = wrong = 0
            for player, assigned in zip(own, identity):
                x = player["location"][0]
                if player.get("actor") or abs(x - 60) < 2 or assigned not in pair:
                    continue
                evaluated += 1
                wrong += (assigned == team) != (x < 60)
            counts[f"{label}_evaluated"], counts[f"{label}_wrong"] = evaluated, wrong
        rows.append({"match_id": match_id, "period": event["period"], "timestamp": event["timestamp"], "event_id": event["id"],
                     "route": resolved.resolved_by, "circular": resolved.resolved_by in EXCLUDED_ROUTES,
                     "offline_unresolved": sum(s == IDENTITY_UNRESOLVED for s in offline[event["id"]].statuses), **counts})
    return rows


def main():
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=7) as pool:
        df = pd.DataFrame([r for out in pool.map(audit, ids, chunksize=2) for r in out])
    df.to_csv(HERE.parent / "kickoff_law8.csv", index=False)
    use = df[~df.circular]
    print("kick-offs with resolved orientation:", len(df), "| non-circular:", len(use), "| by route:", df.route.value_counts().to_dict())
    for label in ("spec", "legacy", "offline", "causal"):
        ev, wr = use[f"{label}_evaluated"].sum(), use[f"{label}_wrong"].sum()
        bad = (use[f"{label}_wrong"] >= 3).sum()
        print(f"{label:8s} players evaluated {ev:6d} wrong half {wr:5d} ({100*wr/ev:.2f}%) | kick-offs with >=3 wrong: {bad}")


if __name__ == "__main__":
    main()
