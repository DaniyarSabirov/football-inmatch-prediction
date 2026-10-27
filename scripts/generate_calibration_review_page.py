"""Generate a static, local HTML review page for ONE match's physical
pitch-orientation calibration decision.

Part of the all-matches feasibility study -- see
docs/all_matches_physical_orientation_feasibility.md. Deliberately NOT a
web application: a human reviewer opens the generated .html file directly
in a browser, looks at two real rendered candidate frames (produced by the
actual, unmodified production renderer -- never a mockup), checks the
listed candidate source links, and records their decision as one new
entry in configs/physical_orientation_calibration.json using the
ready-to-fill JSON snippet the page prints at the bottom.

Why two candidates, not four: the ACTUAL renderer only exposes one binary
choice for physical calibration -- which of the two teams is the
--physical-reference-team (src/team_normalization.py). A third
"horizontal-only" or "vertical-only" variant is not a selectable option
in the real pipeline (those were used only as one-off diagnostics in
docs/full_physical_pitch_orientation_audit.md) -- showing a reviewer a
choice they cannot actually make would be misleading, so this tool
generates exactly the two real choices.

Read-only with respect to raw data and the calibration registry: this
script never writes to configs/physical_orientation_calibration.json
itself -- it only prints the snippet for a human to add by hand (or via a
follow-up, explicitly human-approved edit), so no calibration is ever
silently recorded.

Usage:
    .venv/bin/python scripts/generate_calibration_review_page.py --match-id 3764440
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.animate_statsbomb_360_match import (  # noqa: E402
    JoinedFrame,
    create_landscape_figure,
    determine_team_style,
    get_visual_theme,
    load_and_join_match,
    load_match_metadata_index,
    render_event_frame,
)
from src.orientation_correction import apply_correction, classify_orientation  # noqa: E402
from src.team_normalization import apply_team_normalization_to_event, apply_team_normalization_to_frame  # noqa: E402

OUTPUT_DIR = PROJECT_ROOT / "results" / "physical_orientation" / "review_pages"

# Candidate public source links gathered via WebSearch for the 14-match
# pilot sample (docs/all_matches_physical_orientation_feasibility.md, Part
# 4). These are UNVERIFIED candidates -- titles and URLs only, found via
# text search, never opened or visually inspected by this tool or the
# assistant. Empty for any match not in the pilot sample; the page clearly
# says so rather than inventing sources.
PILOT_CANDIDATE_SOURCES: dict[int, list[dict[str, str]]] = {
    3773403: [
        {"title": "Barcelona 2-1 Athletic (Jan 31, 2021) - ESPN match report", "url": "https://www.espn.com/soccer/report/_/gameId/581994"},
        {"title": "fullmatchsports.cc -- La Liga 2020-2021 full match archive (fan aggregator, unverified legitimacy)", "url": "https://fullmatchsports.cc/full-match-replay/la-liga-2020-2021-full-match/"},
    ],
    3764661: [
        {"title": "Athletic 2-3 Barcelona (Jan 6, 2021) - ESPN match report", "url": "https://www.espn.com/soccer/report/_/gameId/582185"},
        {"title": "footballia.eu -- large free match-video archive, may have this match (not confirmed)", "url": "https://footballia.eu/"},
    ],
    3942226: [
        {"title": "Spain vs Germany Euro 2024 QF highlights - Fox Sports", "url": "https://www.fox.com/watch/clip/fmc-j600rs60l2uoeddb/spain-vs-germany-highlights-uefa-euro-2024-quarterfinals"},
        {"title": "Internet Archive -- Euro 2024 user-uploaded full matches (search required)", "url": "https://archive.org/search?query=euro+2024"},
    ],
    3788776: [
        {"title": "Sweden 3-2 Poland (23 Jun 2021) - ESPN match report", "url": "https://www.espn.co.uk/football/report/_/gameId/560310"},
        {"title": "Internet Archive -- Euro 2020 user-uploaded full matches (search required)", "url": "https://archive.org/search?query=euro+2020"},
    ],
    3802818: [
        {"title": "PSG 5-1 Lorient full match - fullmatchsports.cc (fan aggregator, unverified legitimacy)", "url": "https://fullmatchsports.cc/psg-vs-lorient-full-match-2021-22/"},
        {"title": "PSG 5-1 Lorient full match - fullmatchsreplay.com (fan aggregator, unverified legitimacy)", "url": "https://fullmatchsreplay.com/psg-vs-lorient-full-match-highlights-03-april-2022/"},
    ],
    3835333: [
        {"title": "France 2-1 Belgium (14 Jul 2022) - ESPN match report", "url": "https://www.espn.com/soccer/match/_/gameId/621313/belgium-france"},
    ],
    3835327: [
        {"title": "England 8-0 Norway - UEFA Women's Euro 2022 official report", "url": "https://www.uefa.com/womenseuro/news/0277-1599cbf90e5b-ab827dc81e51-1000--highlights-report-england-progress-after-record-win/"},
    ],
    3857267: [
        {"title": "Ecuador vs Senegal World Cup 2022 - NBC Sports highlights/recap", "url": "https://soccer.nbcsports.com/2022/11/29/ecuador-vs-senegal-video-highlights-recap-analysis/"},
    ],
    3857265: [
        {"title": "Mexico v Poland - official FIFA match centre (watch page)", "url": "https://www.fifa.com/en/watch/5A0FyE3KcUde4ko8lZZilJ"},
    ],
    3877090: [
        {"title": "LAFC vs Inter Miami Sept 4 2023 - ESPN match page", "url": "https://www.espn.com/soccer/match/_/gameId/734497"},
    ],
    3901832: [
        {"title": "Colombia 1-0 Jamaica WWC 2023 - Sky Sports report", "url": "https://www.skysports.com/football/colombia-women-vs-jamaica-women/460883"},
    ],
    3893791: [
        {"title": "Spain 3-0 Costa Rica WWC 2023 - ESPN match page", "url": "https://www.espn.com/soccer/match/_/gameId/622583/costa-rica-spain"},
    ],
    3895052: [
        {"title": "Bayer Leverkusen 3-2 RB Leipzig - Bundesliga official match report", "url": "https://www.bundesliga.com/en/bundesliga/news/rb-leipzig-bayer-leverkusen-match-report-highlights-matchday-19-30632"},
    ],
    3895067: [
        {"title": "Bayer Leverkusen vs Darmstadt full match - footballia.eu", "url": "https://footballia.eu/matches/bayer-leverkusen-sv-darmstadt"},
    ],
}


def pick_representative_event(match, period: int):
    """Pick one usable frame to render: the first Shot in the period if
    one exists (a goal-adjacent moment is the clearest visual landmark
    for a human reviewer), else the temporally-middle usable frame.
    """
    period_frames = [j for j in match.usable_frames if j.event.get("period") == period]
    if not period_frames:
        raise SystemExit(f"No usable 360 frames in period {period} for this match.")
    shots = [j for j in period_frames if j.event.get("type", {}).get("name") == "Shot"]
    return shots[0] if shots else period_frames[len(period_frames) // 2]


def render_candidate(match_id: int, joined: JoinedFrame, team_style, reference_team_id: int, label: str, out_path: Path) -> None:
    orientation_result = classify_orientation(joined.event, joined.frame_360)
    frame_after_a = apply_correction(joined.frame_360, orientation_result.orientation)
    event_final = apply_team_normalization_to_event(joined.event, reference_team_id)
    frame_final = apply_team_normalization_to_frame(frame_after_a, joined.event, reference_team_id)

    from src.animate_statsbomb_360_match import DirectionInference, FrameTiming

    theme = get_visual_theme("dark")
    timing = FrameTiming(match_second=0.0, gap_from_previous=None, gap_to_next=None, display_duration_ms=1500)
    fig, ax = create_landscape_figure(theme)
    render_event_frame(
        ax, match_id, JoinedFrame(event=event_final, frame_360=frame_final), 1, 1,
        timing, "fixed", "camera-overlay", team_style, DirectionInference({}, label), theme,
    )
    fig.savefig(out_path, dpi=100, facecolor=theme.figure_background)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--match-id", type=int, required=True)
    parser.add_argument("--period", type=int, default=1, help="Period to pick the representative event from (default 1).")
    args = parser.parse_args()

    match = load_and_join_match(args.match_id)
    metadata = load_match_metadata_index().get(args.match_id)
    theme = get_visual_theme("dark")
    team_style = determine_team_style(match.usable_frames, theme.default_team_one, theme.default_team_two)

    joined = pick_representative_event(match, args.period)
    event = joined.event

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    match_dir = OUTPUT_DIR / str(args.match_id)
    match_dir.mkdir(parents=True, exist_ok=True)

    team_names = [team_style.team_one, team_style.team_two]
    image_paths = {}
    for team_name in team_names:
        team_id = next(j.event["team"]["id"] for j in match.usable_frames if j.event["team"]["name"] == team_name)
        out_path = match_dir / f"candidate_{team_name.replace(' ', '_').replace('/', '-')}_reference.png"
        render_candidate(args.match_id, joined, team_style, team_id, f"Candidate: {team_name} = reference", out_path)
        image_paths[team_name] = out_path.name

    sources = PILOT_CANDIDATE_SOURCES.get(args.match_id, [])
    sources_html = (
        "".join(f'<li><a href="{s["url"]}" target="_blank">{s["title"]}</a></li>' for s in sources)
        if sources
        else "<li><em>No candidate sources researched yet for this match. "
             "Search for \"&lt;home team&gt; vs &lt;away team&gt; &lt;date&gt; full match\" "
             "to find candidates.</em></li>"
    )

    snippet = {
        "match_id": args.match_id,
        "period_1_physical_reference_team": "<FILL IN: exact team name as shown above>",
        "period_handling_rule": "standard_laws_of_the_game_halftime_and_extra_time_swap",
        "horizontal_orientation_verified": True,
        "vertical_orientation_verified": "<true if the source showed depth/touchline clearly, else false>",
        "vertical_orientation_note": "",
        "source_url": "<FILL IN>",
        "source_description": "<FILL IN>",
        "evidence_landmark": "<FILL IN: e.g. 'X goalkeeper kit color, right-side goal, MM:SS'>",
        "verification_method": "human visual comparison of candidate frame against source footage",
        "confidence": "<high|medium|low>",
        "verification_status": "<verified|partially_verified|unverified|ambiguous>",
        "verified_by": "<FILL IN>",
        "date_verified": "<FILL IN: YYYY-MM-DD>",
        "reviewer_notes": "",
        "report": "docs/all_matches_physical_orientation_feasibility.md",
    }

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Calibration review: match {args.match_id}</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0d1a15; color: #e8efea; max-width: 1400px; margin: 2em auto; }}
h1, h2 {{ color: #f4d35e; }}
.candidates {{ display: flex; gap: 2em; flex-wrap: wrap; }}
.candidate {{ flex: 1; min-width: 500px; }}
.candidate img {{ width: 100%; border: 2px solid #53645d; }}
.candidate h3 {{ text-align: center; }}
table {{ border-collapse: collapse; }}
td, th {{ padding: 4px 10px; border: 1px solid #53645d; text-align: left; }}
pre {{ background: #101f19; padding: 1em; overflow-x: auto; border: 1px solid #53645d; }}
a {{ color: #27a7e8; }}
.warning {{ background: #3a2a0a; border: 1px solid #f4d35e; padding: 1em; margin: 1em 0; }}
</style></head>
<body>
<h1>Physical orientation calibration review: match {args.match_id}</h1>

<div class="warning">
<strong>This page shows only the two choices the renderer can actually make</strong>
(which team is <code>--physical-reference-team</code>) -- not four variants,
since horizontal-only/vertical-only are not real, selectable pipeline
options (see docs/full_physical_pitch_orientation_audit.md). No source
link below has been opened or visually inspected by any automated tool --
you (the reviewer) must open the source yourself and compare it against
the candidate frames below before recording a decision.
</div>

<h2>Match metadata</h2>
<table>
<tr><th>Competition</th><td>{metadata.competition if metadata else '?'}</td></tr>
<tr><th>Season</th><td>{metadata.season if metadata else '?'}</td></tr>
<tr><th>Date</th><td>{metadata.date if metadata else '?'}</td></tr>
<tr><th>Home team</th><td>{metadata.home_team if metadata else '?'}</td></tr>
<tr><th>Away team</th><td>{metadata.away_team if metadata else '?'}</td></tr>
<tr><th>Score</th><td>{metadata.score if metadata else '?'}</td></tr>
<tr><th>Representative event</th><td>{event.get('timestamp')} &middot; period {event.get('period')} &middot; {event.get('type', {}).get('name')} &middot; {event.get('team', {}).get('name')}</td></tr>
</table>

<h2>Candidate orientations (both real, both rendered by the actual production renderer)</h2>
<div class="candidates">
<div class="candidate"><h3>{team_names[0]} = reference</h3><img src="{args.match_id}/{image_paths[team_names[0]]}"></div>
<div class="candidate"><h3>{team_names[1]} = reference</h3><img src="{args.match_id}/{image_paths[team_names[1]]}"></div>
</div>

<h2>Candidate public sources (UNVERIFIED -- found via text search only, never opened)</h2>
<ul>{sources_html}</ul>

<h2>Record your decision here</h2>
<p>After comparing a source above against the two candidates, fill in this
snippet and append it to <code>configs/physical_orientation_calibration.json</code>'s
<code>"calibrations"</code> list:</p>
<pre>{json.dumps(snippet, indent=2)}</pre>

</body></html>
"""
    page_path = OUTPUT_DIR / f"{args.match_id}.html"
    page_path.write_text(html, encoding="utf-8")
    print(f"Review page: {page_path}")
    print(f"Candidate images: {match_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
