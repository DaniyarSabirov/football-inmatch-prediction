"""ML impact of the team-identity findings (investigation only; nothing written
to data/processed). Rebuilds every eligible match's next-event rows IN
MEMORY with the production builder (build_rows_from_match_data, max 10 s,
its CLI default) and joins them with the investigation outputs.

Output: ml_rows_impact.csv.gz (one row per dataset row with exposure flags)
and a printed summary used in ml_impact_analysis.md.
"""

from __future__ import annotations

import ast
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.animate_statsbomb_360_match import available_360_match_ids  # noqa: E402
from src.build_statsbomb_360_dataset import build_rows_from_match_data  # noqa: E402
from src.match_exclusions import load_exclusion_manifest  # noqa: E402

OUT = HERE.parent
DATA = PROJECT_ROOT / "data" / "raw" / "statsbomb" / "data"
FIELDS = ["match_id", "current_event_id", "current_event_type", "orientation", "visible_players",
          "visible_teammates", "visible_opponents", "visible_ambiguous", "actor_x", "actor_y"]


def rows_for(match_id: int) -> list[dict]:
    events = json.loads((DATA / "events" / f"{match_id}.json").read_text(encoding="utf-8"))
    frames = {f["event_uuid"]: f for f in json.loads((DATA / "three-sixty" / f"{match_id}.json").read_text(encoding="utf-8"))}
    try:
        rows = build_rows_from_match_data(match_id, events, frames, 10.0)
    except ValueError:
        return []
    return [{k: r[k] for k in FIELDS} for r in rows]


def main() -> int:
    excluded = load_exclusion_manifest()
    ids = [m for m in available_360_match_ids() if m not in excluded]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = [r for out in pool.map(rows_for, ids, chunksize=2) for r in out]
    df = pd.DataFrame(rows)

    keepers = pd.read_csv(OUT / "keeper_validation.csv.gz")
    keeper_ok = keepers[~keepers.current_drawn_ambiguous].groupby("event_id").current_drawn_consistent.mean() >= 0.5
    own_ok = keepers.groupby("event_id").own_reading_consistent.mean() >= 0.5
    audit = pd.read_csv(OUT / "full_dataset_audit.csv").set_index("event_id")
    switches = pd.read_csv(OUT.parent / "visualization_final_validation" / "team_identity_investigation" / "shared_source_switches.csv")
    refined_events = {e for s in switches.events for e in ast.literal_eval(s)}
    perspective = pd.read_csv(OUT / "record_perspective.csv.gz", usecols=["event_id", "shared_problem_c"]).set_index("event_id")

    df["keeper_evidence"] = df.current_event_id.isin(keeper_ok.index)
    df["keeper_wrong_display"] = df.current_event_id.map(lambda e: (not bool(keeper_ok[e])) if e in keeper_ok.index else False)
    df["root_cause"] = df.current_event_id.map(lambda e: audit.root_cause.get(e, ""))
    df["flag_strength"] = df.current_event_id.map(lambda e: audit.strength.get(e, ""))
    df["display_refined_source_not_in_ml"] = df.current_event_id.isin(refined_events)
    df["shared_problem_c"] = df.current_event_id.map(lambda e: bool(perspective.shared_problem_c.get(e, False)))
    df["uncertain_orientation"] = df.orientation == "UNCERTAIN"
    df["count_swap_delta"] = (df.visible_teammates - df.visible_opponents).abs()
    df.to_csv(OUT / "ml_rows_impact.csv.gz", index=False, compression="gzip")

    real = df.root_cause.isin(["copy_missed_by_problem_c", "derived_record_flags_not_reexpressed",
                               "shared_wrong_perspective", "raw_label_scramble", "record_level_inversion"])
    summary = {
        "rows": len(df), "matches": int(df.match_id.nunique()),
        "rows_with_keeper_evidence": int(df.keeper_evidence.sum()),
        "rows_keeper_verified_wrong_colours": int(df.keeper_wrong_display.sum()),
        "rows_shared_problem_c": int(df.shared_problem_c.sum()),
        "rows_flagged_any": int((df.flag_strength != "").sum()),
        "rows_flagged_strong": int((df.flag_strength == "strong").sum()),
        "rows_by_root_cause": df[df.root_cause != ""].root_cause.value_counts().to_dict(),
        "rows_structural_or_raw_error_classes": int(real.sum()),
        "rows_where_display_refinement_differs_from_ml": int(df.display_refined_source_not_in_ml.sum()),
        "rows_uncertain_orientation_actor_xy_uncorrected_in_ml": int(df.uncertain_orientation.sum()),
        "median_teammate_opponent_gap_in_affected_rows": float(df[df.keeper_wrong_display | real].count_swap_delta.median()),
    }
    (OUT / "ml_impact_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
