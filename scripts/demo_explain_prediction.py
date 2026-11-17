"""Explain ONE next-event prediction of a saved Markov model, step by step (read-only, seconds).

Nothing is trained or written. The script loads:
  - a fitted Markov model saved by experiment 2 (results/markov_hmm_cost_sensitive_v2/models/<tier>_<task>.json),
  - the raw StatsBomb events of ONE development match (refused if it is a final-test match),
  - the stored out-of-fold probabilities (predictions.parquet) to check that the recomputation matches,
  - the stored per-fold cost matrices (cost_matrix_definitions.json).

It prints the prediction context (previous event, current event, pitch third), every backoff level of
the hierarchical Dirichlet smoothing, the probability of all classes, and the uniform / moderate / strong
cost-sensitive decisions with their expected costs.

Usage (PyCharm terminal, project root):
    .venv/bin/python scripts/demo_explain_prediction.py
    .venv/bin/python scripts/demo_explain_prediction.py --model mc1 --task filtered8
    .venv/bin/python scripts/demo_explain_prediction.py --event-id <StatsBomb event UUID>
    .venv/bin/python scripts/demo_explain_prediction.py --match-id 3773386 --minute 12 --event-type Carry

Selecting an event with --match-id / --minute / --event-type
    Candidates are ONLY the rows of the stored out-of-fold predictions (predictions.parquet) for the chosen
    --tier and --task. These are development matches only; final-test matches were never predicted and a
    final-test --match-id is refused before anything is read. Each row is scored by the model of its own
    validation fold (trained on the other folds); nothing is retrained.

    --event-type T  matches the CURRENT event of the prediction row (the last observed event, i.e. the model
                    input), not the true next event. T must belong to the --task vocabulary (case-insensitive).
    --minute M      is the StatsBomb-style match-clock minute of the current event:
                        M = period start offset + floor(period-relative timestamp / 60 s),
                    with offsets period 1 -> 0, period 2 -> 45, extra time 1 -> 90, extra time 2 -> 105.
                    So --minute 0 is the first 60 s, --minute 12 is 12:00-12:59, --minute 50 is the second
                    half's 05:00-05:59. Stoppage time keeps counting (45+2 of the first half is minute 47), so a
                    minute can occur in two periods; both are listed as candidates with their period.
    The filters combine with AND (and with --event-id if given). If exactly one row matches it is explained.
    If several match, they are listed (sorted by match, period, event order) with event IDs and timestamps;
    rerun with --event-id <id> for a deterministic choice (exit code 2). If none match, the reason is printed
    (exit code 1). Without any of these three options the original default/--event-id behaviour is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sequence_models.backoff_markov import BackoffMarkov  # noqa: E402
from src.sequence_models.config_v2 import load_config_v2  # noqa: E402
from src.sequence_models.cost_sensitive import bayes_decisions, expected_costs  # noqa: E402
from src.sequence_models.raw_events import load_match_event_records  # noqa: E402
from src.sequence_models.tasks import build_task_table  # noqa: E402
from src.sequence_models.zones import ZONE_NAMES, zone_of_x  # noqa: E402

EXP2 = PROJECT_ROOT / "results" / "markov_hmm_cost_sensitive_v2"

# StatsBomb match-clock offset (minutes) at the start of each period; 5 = penalty shoot-out
PERIOD_START_MINUTE = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
SELECTION_COLUMNS = ["match_id", "event_id", "fold_id", "prediction_period", "prediction_time",
                     "prediction_event_index", "current_event", "true_next_event"]
MAX_LISTED_CANDIDATES = 30
EXIT_AMBIGUOUS = 2


class SelectionError(Exception):
    """No (or no permitted) prediction row matches; the message explains why."""


def timestamp_seconds(ts: str) -> float:
    """Period-relative StatsBomb timestamp 'HH:MM:SS.mmm' -> seconds."""
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def match_clock_minute(period: int, ts: str) -> int:
    """StatsBomb-style match minute: period start offset + whole minutes elapsed in the period."""
    if int(period) not in PERIOD_START_MINUTE:
        raise ValueError(f"unknown period {period}")
    return PERIOD_START_MINUTE[int(period)] + int(timestamp_seconds(ts) // 60)


def canonical_event_type(name: str, vocab: list[str], task: str) -> str:
    lookup = {v.lower(): v for v in vocab}
    if name.lower() not in lookup:
        raise SelectionError(f"event type {name!r} is not in the {task} vocabulary, so no prediction row can have "
                             f"it as current event. Valid types: {', '.join(vocab)}")
    return lookup[name.lower()]


def check_match_permitted(match_id: int, split: dict, tier: str) -> None:
    """Refuse final-test matches before reading anything; explain unknown or out-of-tier matches."""
    if match_id in set(split["final_test_match_ids"]):
        raise SelectionError(f"refused: match {match_id} is a final-test match (held out, "
                             f"{split.get('final_test_status', 'untouched')}); only development predictions are used")
    if match_id not in set(split["development_match_ids"]):
        raise SelectionError(f"match {match_id} is not in the development set of experiment 2 "
                             f"({len(split['development_match_ids'])} matches), so it has no stored predictions")
    tier_ids = set(split.get(tier, {}).get("match_ids", split["development_match_ids"]))
    if match_id not in tier_ids:
        raise SelectionError(f"match {match_id} is a development match but not part of {tier} "
                             f"({len(tier_ids)} matches); try another --tier")


def select_candidates(rows, *, event_id=None, match_id=None, minute=None, event_type=None):
    """Filter stored prediction rows (DataFrame with SELECTION_COLUMNS) by the given criteria (AND).

    Returns the matching rows sorted deterministically, or raises SelectionError explaining which criterion
    removed the remaining rows.
    """
    df = rows.copy()
    df["clock_minute"] = [match_clock_minute(p, t) for p, t in zip(df["prediction_period"], df["prediction_time"])]
    scope = "the stored predictions of this tier/task"
    if event_id is not None:
        df = df[df["event_id"] == event_id]
        if df.empty:
            raise SelectionError(f"event {event_id} is not among {scope}. Only development events whose current and "
                                 f"next events are in the task vocabulary have a prediction row")
    if match_id is not None:
        df = df[df["match_id"] == match_id]
        if df.empty:
            raise SelectionError(f"match {match_id} has no rows in {scope}"
                                 + (f" containing event {event_id}" if event_id else ""))
        scope = f"match {match_id}"
    if minute is not None:
        if minute < 0:
            raise SelectionError("--minute must be >= 0")
        by_type = df[df["current_event"] == event_type] if event_type is not None else df
        hit = df[df["clock_minute"] == minute]
        if hit.empty:
            avail = sorted(set(df["clock_minute"]))
            near = sorted(avail, key=lambda m: (abs(m - minute), m))[:5]
            raise SelectionError(f"no prediction row in {scope} has its current event in match minute {minute} "
                                 f"(minutes with rows: {avail[0]}-{avail[-1]}; nearest: {sorted(near)}). Events of "
                                 f"types outside the task vocabulary, and the last event of a sequence, have no row")
        df = hit
        if event_type is not None and df[df["current_event"] == event_type].empty:
            types = df["current_event"].value_counts()
            minutes = sorted(set(by_type["clock_minute"]))
            raise SelectionError(
                f"{scope}, minute {minute} has {len(df)} prediction row(s), but none with current event "
                f"{event_type} (present: {', '.join(f'{k} {v}' for k, v in types.items())}). "
                + (f"{event_type} occurs in {scope} at minutes {minutes}" if minutes
                   else f"{event_type} never occurs as current event in {scope}"))
    if event_type is not None:
        df = df[df["current_event"] == event_type]
        if df.empty:
            raise SelectionError(f"no prediction row in {scope} has current event {event_type}")
    return df.sort_values(["match_id", "prediction_period", "prediction_event_index", "event_id"]).reset_index(drop=True)


def format_candidates(df, limit: int = MAX_LISTED_CANDIDATES) -> str:
    lines = [f"{len(df)} prediction rows match; rerun with --event-id <id> to choose one deterministically"
             + (f" (first {limit} shown; add --match-id/--minute/--event-type to narrow)" if len(df) > limit else "")
             + ":",
             f"  {'event_id':<36}  {'match_id':>8}  fold  per  {'timestamp':<12}  min  {'current':<15} true next"]
    for r in df.head(limit).itertuples(index=False):
        lines.append(f"  {r.event_id:<36}  {r.match_id:>8}  {r.fold_id:>4}  {r.prediction_period:>3}  "
                     f"{r.prediction_time:<12}  {r.clock_minute:>3}  {r.current_event:<15} {r.true_next_event}")
    return "\n".join(lines)


def describe_key(level: tuple[str, ...], ctx: dict[str, int], vocab: list[str]) -> str:
    names = {"cur": lambda v: f"current={vocab[v]}",
             "prev": lambda v: "previous=START" if v == len(vocab) else f"previous={vocab[v]}",
             "zone": lambda v: "zone=UNKNOWN" if v == len(ZONE_NAMES) else f"zone={ZONE_NAMES[v]}"}
    return ", ".join(names[f](ctx[f]) for f in level)


def resolve_event_id(args, vocab: list[str], split: dict, data) -> str:
    """New selection path (--match-id / --minute / --event-type): returns the single matching event_id."""
    event_type = canonical_event_type(args.event_type, vocab, args.task) if args.event_type is not None else None
    if args.match_id is not None:
        check_match_permitted(args.match_id, split, args.tier)
    flt = (pc.field("tier") == args.tier) & (pc.field("task") == args.task)
    if args.match_id is not None:
        flt = flt & (pc.field("match_id") == args.match_id)
    if args.event_id:
        flt = flt & (pc.field("event_id") == args.event_id)
    rows = data.to_table(columns=SELECTION_COLUMNS, filter=flt).to_pandas()
    rows = rows[~rows["match_id"].isin(set(split["final_test_match_ids"]))]  # defensive; none are stored
    cand = select_candidates(rows, event_id=args.event_id, match_id=args.match_id, minute=args.minute,
                             event_type=event_type)
    if len(cand) > 1:
        print(format_candidates(cand))
        raise SystemExit(EXIT_AMBIGUOUS)
    return str(cand.at[0, "event_id"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tier", default="tier_b", choices=["tier_a", "tier_b"])
    parser.add_argument("--task", default="scenarioD15", choices=["filtered8", "scenarioD15"])
    parser.add_argument("--model", default="mc2_zone", choices=["mc1", "mc2", "mc1_zone", "mc2_zone"])
    parser.add_argument("--event-id", default=None, help="default: first fold-0 row whose true next event is a Shot")
    parser.add_argument("--match-id", type=int, default=None,
                        help="restrict to one development match (final-test matches are refused)")
    parser.add_argument("--minute", type=int, default=None,
                        help="match-clock minute of the CURRENT event: period offset (0/45/90/105) + "
                             "floor(period timestamp in minutes); stoppage time keeps counting")
    parser.add_argument("--event-type", default=None,
                        help="type of the CURRENT (last observed) event, from the --task vocabulary")
    args = parser.parse_args(argv)

    config = load_config_v2()
    split = json.loads((EXP2 / "split_manifest.json").read_text(encoding="utf-8"))
    vocab = list(config.tasks[args.task].vocabulary)

    # 1. pick the prediction row from the stored out-of-fold predictions (only a few columns are read)
    data = ds.dataset(EXP2 / "predictions.parquet")
    flt = (pc.field("tier") == args.tier) & (pc.field("task") == args.task)
    if any(v is not None for v in (args.match_id, args.minute, args.event_type)):
        try:
            flt = flt & (pc.field("event_id") == resolve_event_id(args, vocab, split, data))
        except SelectionError as exc:
            raise SystemExit(f"no event selected: {exc}") from None
    elif args.event_id:
        flt = flt & (pc.field("event_id") == args.event_id)
    else:
        flt = flt & (pc.field("fold_id") == 0) & (pc.field("true_next_event") == "Shot")
    cols = ["match_id", "event_id", "fold_id", "sequence_id", "prediction_period", "prediction_time",
            "current_event", "true_next_event"] + [f"p__{args.model}__{c}" for c in vocab]
    rows = data.to_table(columns=cols, filter=flt).slice(0, 1).to_pylist()
    if not rows:
        raise SystemExit("no matching evaluation row")
    row = rows[0]
    match_id, fold = int(row["match_id"]), int(row["fold_id"])
    if match_id in set(split["final_test_match_ids"]):
        raise SystemExit("refused: final-test match")

    # 2. rebuild the sequence of this ONE match and the context of the event
    import pandas as pd
    events = pd.DataFrame(load_match_event_records(match_id))
    table = build_task_table(events, config.tasks[args.task]).merge(events[["event_id", "location_x"]], on="event_id")
    i = int(np.flatnonzero(table["event_id"].to_numpy() == row["event_id"])[0])
    cur = int(table.at[i, "symbol_id"])
    same_seq = i > 0 and table.at[i - 1, "sequence_id"] == table.at[i, "sequence_id"]
    prev = int(table.at[i - 1, "symbol_id"]) if same_seq else len(vocab)
    x = float(table.at[i, "location_x"])
    zone = int(zone_of_x(np.array([x]))[0])
    zone_ctx = len(ZONE_NAMES) if zone < 0 else zone
    ctx = {"cur": cur, "prev": prev, "zone": zone_ctx}
    print(f"\nMatch {match_id}, period {row['prediction_period']}, {row['prediction_time']} (event {row['event_id']})")
    print(f"  previous event : {'START of period' if prev == len(vocab) else vocab[prev]}")
    print(f"  current event  : {vocab[cur]} at x = {x:.1f} (acting team attacks towards x = 120) -> {ZONE_NAMES[zone]}")
    print(f"  true next event: {row['true_next_event']}   (used only for evaluation, never as input)")
    print(f"  model          : {args.model}, trained on the OTHER folds of {args.tier} (validation fold {fold})")

    # 3. step through the hierarchical Dirichlet backoff
    model = BackoffMarkov.from_dict(json.loads((EXP2 / "models" / f"{args.tier}_{args.task}.json").read_text())["markov"]
                                    [f"{args.model}|fold{fold}"])
    v, beta = model.n_symbols, model.concentration
    p = (model.base_counts + model.base_alpha) / (model.base_counts.sum() + v * model.base_alpha)
    top = lambda q: ", ".join(f"{vocab[k]} {q[k]:.3f}" for k in np.argsort(q)[::-1][:4])  # noqa: E731
    print(f"\nStep 0  marginal next-event frequency (Laplace alpha={model.base_alpha}): {top(p)}")
    for step, (level, counts) in enumerate(zip(reversed(model.levels), reversed(model.level_counts)), start=1):
        key = model._key(level, {f: np.array([ctx[f]]) for f in level})[0]
        c = counts[key]
        p = (c + beta * p) / (c.sum() + beta)
        print(f"Step {step}  context ({describe_key(level, ctx, vocab)}): seen {int(c.sum()):,} times in training;")
        print(f"        p = (counts + {beta:g} * previous step) / ({int(c.sum()):,} + {beta:g})  ->  {top(p)}")

    stored = np.array([row[f"p__{args.model}__{k}"] for k in vocab], dtype=float)
    print(f"\nAll {v} classes (recomputed | stored in predictions.parquet):")
    for k in np.argsort(p)[::-1]:
        print(f"  {vocab[k]:<15} {p[k]:.4f} | {stored[k]:.4f}")
    print(f"  max |recomputed - stored| = {np.abs(p - stored).max():.2e} (stored as float32)")

    # 4. decisions: argmax vs cost-sensitive (Bayes) rule with the stored fold cost matrices
    costs = json.loads((EXP2 / "cost_matrix_definitions.json").read_text())["per_tier"][args.tier][args.task]["folds"][f"fold{fold}"]
    print("\nDecisions (probabilities above are unchanged by the decision rule):")
    for scheme in ("uniform", "moderate", "strong"):
        c = np.array(costs[scheme]["matrix"])
        risk = expected_costs(p[None, :], c)[0]
        d = int(bayes_decisions(p[None, :], c)[0])
        best = ", ".join(f"{vocab[a]} {risk[a]:.3f}" for a in np.argsort(risk)[:3])
        print(f"  {scheme:<8}: predict {vocab[d]:<14} lowest expected costs: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
