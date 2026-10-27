"""Build the broadcast-orientation manual-review queue and review page.

Reads results/physical_orientation/final_orientation_audit/per_match_audit.json
(run scripts/audit_broadcast_orientation.py first) and selects the matches
that should be checked against real footage:

  1. every match NOT classified high-confidence or verified (must review);
  2. the extra-time matches whose period 3 continues at period 2's ends
     (unusual, worth confirming);
  3. shootout matches (the shared-frame model);
  4. the lowest-agreement inferred matches;
  5. one deterministic random inferred match per competition/season
     (stratified validation sample of the automatic method).

For each queued match it renders, with the production renderer, a period-1
and a period-2 snapshot in the PROPOSED orientation (camera touchline at
the bottom) and in the 180-degree ALTERNATIVE, and writes review.html: one
match at a time, keyboard/radio decisions, evidence fields, saved in the
browser and exported as a JSON file. scripts/import_orientation_reviews.py
turns that export into registry entries -- nobody edits the registry JSON by
hand.

Usage:
    .venv/bin/python scripts/build_orientation_review_queue.py [--sample-per-stratum 1] [--match-id N ...]
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    DirectionInference,
    apply_display_plan_to_match,
    apply_orientation_correction_to_match,
    build_timing_plan,
    compute_orientation_results,
    compute_shared_group_info,
    create_landscape_figure,
    determine_team_style,
    display_orientation_results,
    get_visual_theme,
    infer_broadcast_orientation_for_match,
    load_and_join_match,
    match_team_ids,
    orientation_provenance_text,
    render_event_frame,
)
from src.broadcast_orientation import build_display_plan  # noqa: E402
from src.orientation_correction import ORIENTATION_ROTATED_180, apply_correction  # noqa: E402
from src.physical_orientation_calibration import load_period_reviews  # noqa: E402
from src.team_normalization import resolve_reference_team_id, rotate_event_180  # noqa: E402

AUDIT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "final_orientation_audit"
REVIEW_DIR = AUDIT_DIR / "review"


def select_queue(rows: list[dict], sample_per_stratum: int, extra_ids: list[int]) -> list[tuple[int, str]]:
    eligible = [r for r in rows if r["category"] != "excluded_insufficient_data"]
    queue: dict[int, str] = {}

    def add(match_id: int, why: str) -> None:
        queue.setdefault(match_id, why)

    for r in eligible:
        if r["category"] not in ("high_confidence_inferred", "physically_verified", "user_reviewed_and_derived"):
            add(r["match_id"], f"must review: {r['category']} ({r['reason']})")
    for r in eligible:
        refs = r.get("camera_reference_by_period") or {}
        if "3" in refs and refs.get("3") == refs.get("2"):
            add(r["match_id"], "extra time: period 3 inferred at period 2's ends")
    for r in eligible:
        if "5" in (r.get("camera_period_evidence") or {}):
            add(r["match_id"], "penalty shootout (shared-frame model)")
    inferred = [r for r in eligible if r["category"] == "high_confidence_inferred"]
    lowest = sorted(inferred, key=lambda r: min(e["agreement"] for p, e in r["camera_period_evidence"].items() if p != "5"))
    for r in lowest[:5]:
        add(r["match_id"], "lowest camera-side agreement among inferred matches")
    strata: dict[tuple, list[dict]] = {}
    for r in inferred:
        strata.setdefault((r.get("competition"), r.get("season")), []).append(r)
    rng = random.Random(20261023)
    for key in sorted(strata, key=str):
        for r in rng.sample(strata[key], min(sample_per_stratum, len(strata[key]))):
            add(r["match_id"], f"stratified validation sample: {key[0]} {key[1]}")
    for match_id in extra_ids:
        add(match_id, "requested explicitly")
    return list(queue.items())


def youtube_search_url(row: dict) -> str:
    query = f"{row.get('home_team')} vs {row.get('away_team')} {row.get('date', '')[:4]} {row.get('competition')} highlights"
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)


def render_snapshots(match_id: int, row: dict, reviews: dict) -> dict:
    raw = load_and_join_match(match_id)
    results = compute_orientation_results(raw)
    periods = sorted({j.event["period"] for j in raw.usable_frames})
    team_ids = match_team_ids(raw)
    events = [j.event for j in raw.usable_frames]
    inference = infer_broadcast_orientation_for_match(raw, results)
    plan = build_display_plan(
        inference, team_ids, periods,
        fallback_period_1_reference=resolve_reference_team_id(events),
        confirmed_references={p: resolve_reference_team_id(events, r.display_reference_team) for p, r in reviews.items()},
    )
    used, _ = display_orientation_results(raw, results, inference, plan)
    display = apply_display_plan_to_match(apply_orientation_correction_to_match(raw, used), plan)
    by_id = {j.event["id"]: j for j in display.usable_frames}
    names = {j.event["team"]["id"]: j.event["team"]["name"] for j in raw.usable_frames}
    style = determine_team_style(raw.usable_frames, "#4aa8ff", "#ff5a5f")
    shared = compute_shared_group_info(raw)
    raw_events = {j.event["id"]: j.event for j in raw.usable_frames}
    theme = get_visual_theme("dark")
    out_dir = REVIEW_DIR / str(match_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    images = {}
    for period_key, event_id in sorted((row.get("representative_event_ids") or {}).items()):
        period = int(period_key)
        if period not in (1, 2) or event_id not in by_id:
            continue
        proposed = by_id[event_id]
        alternative = type(proposed)(
            event=rotate_event_180(proposed.event),
            frame_360=apply_correction(proposed.frame_360, ORIENTATION_ROTATED_180),
        )
        directions = {names[t]: d for t, d in (plan.attacking_directions(period) or {}).items()}
        for label, joined, dirs in (("proposed", proposed, directions), ("alternative", alternative, {k: -v for k, v in directions.items()})):
            figure, ax = create_landscape_figure(theme)
            timing = build_timing_plan([joined], "fixed", 250, 1.0, 80, 1500)[0]
            render_event_frame(
                ax, match_id, joined, 1, 1, timing, "fixed", "camera-overlay", style,
                DirectionInference(dirs, "", orientation_provenance_text(plan, period) if label == "proposed" else "ALTERNATIVE: proposed view rotated 180 degrees"),
                theme, False, True, shared, raw_events,
            )
            path = out_dir / f"p{period}_{label}.png"
            figure.savefig(path, dpi=55, facecolor=theme.figure_background)
            plt.close(figure)
            images[f"p{period}_{label}"] = path.relative_to(REVIEW_DIR).as_posix()
    return {
        "images": images,
        "proposed_period_1_reference": names.get(plan.transforms[1].reference_team_id) if 1 in plan.transforms else None,
        "alternative_period_1_reference": (
            names[next(t for t in team_ids if t != plan.transforms[1].reference_team_id)] if 1 in plan.transforms else None
        ),
        "proposed_reference_by_period": {
            str(p): names[t.reference_team_id] for p, t in plan.transforms.items() if t.reference_team_id is not None
        },
        "alternative_reference_by_period": {
            str(p): names[next(x for x in team_ids if x != t.reference_team_id)]
            for p, t in plan.transforms.items() if t.reference_team_id is not None
        },
        "source_by_period": {str(p): t.source for p, t in plan.transforms.items()},
        "recorded_reviews": {
            str(p): f"{r.decision or 'confirmed'} ({r.evidence_type}, {r.reviewed_by}, {r.date_reviewed})"
            for p, r in reviews.items()
        },
        "plan": plan.explanation,
        "provenance": plan.provenance,
    }


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Orientation Review Queue</title>
<style>
:root{--bg:#f6f7f9;--fg:#1d232a;--muted:#5d6670;--card:#fff;--line:#d9dde2;--accent:#1f6feb}
@media (prefers-color-scheme:dark){:root{--bg:#0f1317;--fg:#e6e9ec;--muted:#9aa4ad;--card:#171c21;--line:#2a3138;--accent:#58a6ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding:16px}
header{display:flex;flex-wrap:wrap;gap:8px;align-items:center;justify-content:space-between}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media (max-width:760px){.grid{grid-template-columns:1fr}}
figure{margin:0}figure img{width:100%;border-radius:6px;border:1px solid var(--line)}
figcaption{color:var(--muted);font-size:13px;margin:4px 0}
label{display:block;margin:6px 0}input[type=text],textarea{width:100%;box-sizing:border-box;padding:6px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{padding:7px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.muted{color:var(--muted)}a{color:var(--accent)}
</style></head><body><main>
<header><h1 style="margin:0;font-size:20px">Broadcast orientation review</h1>
<div><button id="prev">&larr; Prev</button> <span id="pos"></span> <button id="next">Next &rarr;</button>
<button class="primary" id="export">Export decisions</button></div></header>
<p class="muted">Each match: compare the PROPOSED view (camera touchline at the bottom) and the 180&deg; ALTERNATIVE with real footage.
Use scoreboard clock, period, kits and goalkeepers to identify the half; ignore replays and reverse angles. Decisions are kept in this browser
and exported as a JSON file for <code>scripts/import_orientation_reviews.py</code>. Keys: <b>1</b> proposed, <b>2</b> alternative, <b>3</b> can't tell, &larr;/&rarr; navigate.</p>
<div id="app"></div>
</main>
<script id="queue" type="application/json">__QUEUE__</script>
<script>
const queue = JSON.parse(document.getElementById('queue').textContent);
const KEY = 'orientation-review-v1';
let store = {}; try { store = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
let i = 0;
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(store)); } catch (e) {} };
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
function fig(m, key, caption) { return m.images[key] ? `<figure><figcaption>${caption}</figcaption><img loading="lazy" src="${m.images[key]}" alt="${caption}"></figure>` : `<p class="muted">${caption}: no snapshot</p>`; }
function render() {
  const m = queue[i], d = store[m.match_id] || {};
  document.getElementById('pos').textContent = `${i + 1} / ${queue.length}`;
  const field = (k, label, ph) => `<label>${label}<input type="text" data-k="${k}" value="${esc(d[k])}" placeholder="${ph}"></label>`;
  const radio = (v, text) => `<label><input type="radio" name="decision" value="${v}" ${d.decision === v ? 'checked' : ''}> ${text}</label>`;
  document.getElementById('app').innerHTML = `
  <div class="card"><h2 style="margin:0 0 4px;font-size:18px">${esc(m.home_team)} vs ${esc(m.away_team)} <span class="muted">(${m.match_id})</span></h2>
  <div class="muted">${esc(m.competition)} ${esc(m.season)} &middot; ${esc(m.date)} &middot; score ${esc(m.score)} &middot; keepers: ${esc(m.goalkeepers)}</div>
  <div>Why queued: <b>${esc(m.why)}</b> &middot; status ${esc(m.category)} &middot; plan provenance ${esc(m.provenance)}</div>
  <div class="muted">Plan: ${esc(m.plan)}</div>
  <div>Recorded reviews: ${Object.keys(m.recorded_reviews || {}).length ? Object.entries(m.recorded_reviews).map(([p, t]) => `P${p} ${esc(t)}`).join('; ') : 'none'}</div>
  <div><a href="${m.search_url}" target="_blank" rel="noopener">Search highlights (YouTube search &mdash; not a verified source)</a></div></div>
  <div class="card grid">${fig(m,'p1_proposed','Period 1 &mdash; PROPOSED')}${fig(m,'p1_alternative','Period 1 &mdash; ALTERNATIVE (180&deg;)')}
  ${fig(m,'p2_proposed','Period 2 &mdash; PROPOSED')}${fig(m,'p2_alternative','Period 2 &mdash; ALTERNATIVE (180&deg;)')}</div>
  <div class="card"><b>Decision</b>
  ${radio('proposed', `Broadcast matches PROPOSED (period-1 team attacking right: ${esc(m.proposed_period_1_reference)})`)}
  ${radio('alternative', `Broadcast matches ALTERNATIVE (period-1 team attacking right: ${esc(m.alternative_period_1_reference)})`)}
  ${radio('cannot_tell', 'Cannot tell from available footage')}
  <div>Periods this decision covers (leave all unticked for a whole-match decision):
  ${Object.keys(m.proposed_reference_by_period || {}).map(p => `<label style="display:inline-block;margin-right:12px"><input type="checkbox" data-period="${p}" ${(d.periods || []).includes(+p) ? 'checked' : ''}> P${p}${m.images['p' + p + '_proposed'] ? '' : ' (no snapshot)'}</label>`).join('')}</div>
  <label>Evidence type <select data-k="evidence_type">
    <option value="user_review_of_renders" ${d.evidence_type !== 'independent_video' ? 'selected' : ''}>My review of these renders (not independent video)</option>
    <option value="independent_video" ${d.evidence_type === 'independent_video' ? 'selected' : ''}>Independent video I watched (needs URL + clock)</option></select></label>
  ${field('source_url','Source URL (the actual video you watched)','https://...')}
  ${field('video_timestamp','Video timestamp','e.g. 03:12')}
  ${field('match_clock','Match clock and period shown on the scoreboard','e.g. 2nd half 87:43')}
  ${field('goal_side_evidence','Goal-side evidence','e.g. RUS keeper in left goal, scoreboard BEL 2-0 RUS')}
  ${field('camera_side_evidence','Camera-side evidence','e.g. main camera, team benches near side')}
  <label>Notes<textarea data-k="notes" rows="2">${esc(d.notes)}</textarea></label>
  <label>Reviewer<input type="text" data-k="reviewer" value="${esc(d.reviewer)}"></label></div>`;
  document.querySelectorAll('input[name=decision]').forEach(r => r.onchange = () => { upd('decision', r.value); });
  document.querySelectorAll('[data-k]').forEach(el => el.oninput = () => upd(el.dataset.k, el.value));
  document.querySelectorAll('[data-period]').forEach(el => el.onchange = () => upd('periods',
    [...document.querySelectorAll('[data-period]:checked')].map(c => +c.dataset.period)));
}
function upd(k, v) { const m = queue[i]; store[m.match_id] = {...(store[m.match_id] || {}), [k]: v, match_id: m.match_id,
  proposed_period_1_reference: m.proposed_period_1_reference, alternative_period_1_reference: m.alternative_period_1_reference,
  proposed_reference_by_period: m.proposed_reference_by_period, alternative_reference_by_period: m.alternative_reference_by_period,
  reviewed_at: new Date().toISOString()}; save(); }
document.getElementById('prev').onclick = () => { i = (i + queue.length - 1) % queue.length; render(); };
document.getElementById('next').onclick = () => { i = (i + 1) % queue.length; render(); };
document.getElementById('export').onclick = () => {
  const blob = new Blob([JSON.stringify(Object.values(store), null, 1)], {type: 'application/json'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'orientation_review_decisions.json'; a.click();
};
document.addEventListener('keydown', e => {
  if (['INPUT','TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'ArrowRight') document.getElementById('next').click();
  if (e.key === 'ArrowLeft') document.getElementById('prev').click();
  const map = {'1':'proposed','2':'alternative','3':'cannot_tell'};
  if (map[e.key]) { upd('decision', map[e.key]); render(); }
});
render();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample-per-stratum", type=int, default=1)
    parser.add_argument("--match-id", type=int, nargs="*", default=[])
    args = parser.parse_args()

    rows = {r["match_id"]: r for r in json.loads((AUDIT_DIR / "per_match_audit.json").read_text(encoding="utf-8"))}
    period_reviews = load_period_reviews()
    queue = []
    for match_id, why in select_queue(list(rows.values()), args.sample_per_stratum, args.match_id):
        row = rows[match_id]
        entry = {
            "match_id": match_id, "why": why,
            **{k: row.get(k) for k in ("competition", "season", "date", "home_team", "away_team", "score", "category")},
            "goalkeepers": "; ".join(f"{t}: {', '.join(n)}" for t, n in (row.get("goalkeepers") or {}).items()),
            "search_url": youtube_search_url(row),
            **render_snapshots(match_id, row, period_reviews.get(match_id, {})),
        }
        queue.append(entry)
        print(f"queued {match_id}: {why}")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    (AUDIT_DIR / "manual_review_queue.json").write_text(json.dumps(queue, indent=1), encoding="utf-8")
    payload = json.dumps(queue).replace("</", "<\\/")
    (REVIEW_DIR / "review.html").write_text(PAGE.replace("__QUEUE__", payload), encoding="utf-8")
    print(f"{len(queue)} matches -> {REVIEW_DIR / 'review.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
