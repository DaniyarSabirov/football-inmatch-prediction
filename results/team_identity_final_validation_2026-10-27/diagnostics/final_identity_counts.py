"""Final dataset-wide identity totals (offline and causal) with the final code.
Output: ../final_identity_counts.json"""
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402
from src.orientation_resolution import match_team_ids, resolve_match_orientation  # noqa: E402
from src.team_identity import TEAM_IDENTITY_AMBIGUOUS, resolve_match_team_identity  # noqa: E402

DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"


def count(match_id):
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    mo = resolve_match_orientation(match_id, events, frames)
    recs = mo.records.usable_frames
    names = {r.event["team"]["id"]: r.event["team"]["name"] for r in recs}
    pair = tuple(names[t] for t in match_team_ids(mo.records))
    inputs = ([(r.event, r.frame_360) for r in recs], [x.orientation for x in mo.resolved], mo.problem_a, pair, events)
    c = Counter()
    for mode, ids in (("offline", resolve_match_team_identity(*inputs)), ("causal", resolve_match_team_identity(*inputs, causal=True))):
        for f in ids.values():
            c[f"{mode}_snapshots"] += 1
            c[f"{mode}_snapshots_with_unresolved"] += f.has_unresolved
            for team, legacy, status, reason in zip(f.teams, f.legacy_teams, f.statuses, f.reasons):
                c[f"{mode}_points"] += 1
                c[f"{mode}_status_{status}"] += 1
                if reason:
                    c[f"{mode}_reason_{reason}"] += 1
                if team != TEAM_IDENTITY_AMBIGUOUS and legacy != TEAM_IDENTITY_AMBIGUOUS and team != legacy:
                    c[f"{mode}_changed_vs_original_pipeline"] += 1
                    c[f"{mode}_changed_vs_original_by_{status}"] += 1
                if team == TEAM_IDENTITY_AMBIGUOUS and legacy != TEAM_IDENTITY_AMBIGUOUS:
                    c[f"{mode}_original_definite_now_unresolved"] += 1
    return c


if __name__ == "__main__":
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    total = Counter()
    with ProcessPoolExecutor(max_workers=7) as pool:
        for c in pool.map(count, ids, chunksize=2):
            total.update(c)
    (HERE.parent / "final_identity_counts.json").write_text(json.dumps(dict(sorted(total.items())), indent=1), encoding="utf-8")
    print(json.dumps(dict(sorted(total.items())), indent=1))
