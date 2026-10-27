# Feasibility Study: Physical Pitch Orientation Calibration for All Eligible Matches

Date: 2026-10-22. Builds directly on the Spain–Croatia verified case
(`docs/full_physical_pitch_orientation_audit.md`,
`docs/physical_pitch_orientation_validation.md`). No raw data was
modified, the ML dataset builder is untouched, no existing verified
calibration was overwritten, and nothing was committed or staged.

**Critical constraint, stated up front and honored throughout this
report**: the assistant producing this report has no video-playback or
frame-inspection capability in this environment. `WebSearch`/`WebFetch`
return text/HTML only. Every claim below distinguishes what was
**searched for** (candidate source discovery, fully automatable) from
what was **visually verified** (requires a human watching footage —
not performed by the assistant for any match in this report beyond what
was already verified in the prior session for Spain–Croatia).

---

## Executive answers (asked for at the end of the task; given first for visibility)

| Question | Answer |
|---|---|
| Is manually calibrating all 417 matches realistic? | Yes, as a multi-week effort (see §3 estimate: ~70–210 human-hours depending on footage accessibility), not as a quick task. Realistic if spread over time and not attempted as one uncontrolled pass. |
| Can Claude assist with source discovery? | Yes — demonstrated in §4: 14/14 pilot matches got at least one candidate source link from a single search each, fully automatable at scale. Claude cannot perform the visual verification step itself. |
| How many matches could plausibly be verified? | Unknown precisely without attempting all 417; the 14-match pilot suggests the large majority have *some* findable candidate source, but accessibility (paywalls, upload gaps) will leave an unverifiable residual, plausibly 10–30% based on this small sample (not a statistically reliable estimate — see §4 caveats). |
| What evidence is required for a scientifically defensible calibration? | An identifiable, reproducible visual landmark (team kit color at a known goal, scoreline, or similar) cross-referenced against an independent, citable source (URL + timestamp), recorded with method, confidence, and verifier identity — exactly the schema in §5 and the pattern already used for match 3930160. |
| What is the fastest reliable workflow? | Per-match: generate the two-candidate HTML review page (§7, already built, <1s/match) → human finds one source → human compares → human fills the printed JSON snippet → append to the registry. No faster workflow is justified without first validating this one at modest scale. |
| Should we implement the calibration registry now? | **Already implemented** (this session) — schema, loader, auto-apply logic, and the one verified entry are in place and tested. |
| Should we proceed with all 417 matches or start with a smaller validated subset? | **Start with a validated subset.** Recommend running the §7 workflow on a further 20–30 matches (prioritized by the thesis's actual analysis needs, per §8) before deciding whether a full 417-match effort is worthwhile. |

---

## Part 1 — Match inventory

**Script**: `scripts/build_match_orientation_inventory.py` (read-only;
reuses `load_and_join_match`, `resolve_reference_team_id`, and the
existing match-metadata loader — no new data-loading logic invented).

**Output**: `results/physical_orientation/match_inventory.csv` — one row
per of the 426 raw matches (local three-sixty file present), with:
`match_id`, `eligible`, `exclusion_reason`, `competition`, `season`,
`date`, `home_team`, `away_team`, `score`, `stadium`, `periods_present`,
`usable_360_snapshots`, `automatic_reference_team` (the existing,
non-physical default — included for transparency, never confused with a
physical claim), and the four calibration-status columns (populated from
`configs/physical_orientation_calibration.json` if present).

### Summary (computed directly, 0 load errors across all 426 matches)

| Metric | Value |
|---|---|
| Raw matches (local 360 data) | 426 |
| Eligible matches | 417 |
| Physically verified (registry) | 1 (match 3930160) |
| Distinct competitions among eligible matches | 8 |
| Distinct stadiums | 109 |
| Distinct teams | 150 (142 appear in >1 eligible match) |

| Competition | Eligible matches |
|---|---:|
| UEFA Euro (2020 and 2024 seasons combined) | 102 |
| FIFA World Cup | 64 |
| Women's World Cup | 64 |
| UEFA Women's Euro | 61 |
| Ligue 1 | 57 |
| La Liga | 35 |
| 1. Bundesliga | 31 |
| Major League Soccer | 3 |

### Which matches are easier to calibrate, and shared-evidence potential

- **142 of 150 teams appear in more than one eligible match** — but this
  does **not** mean one verified frame per team suffices: the renderer's
  reference-team choice (and therefore the calibration needed) is a
  **per-match** fact (which team's raw convention happens to match that
  specific match's specific broadcast camera), not a per-team constant —
  confirmed by this investigation's own prior finding that the automatic
  pick correlates with "whichever team kicks off," not team identity.
  Shared team identity reduces *search* effort (the same team's kit
  colors/known landmarks are reusable context for a human reviewer) but
  does **not** let one calibration propagate across different matches of
  the same team.
- **109 distinct stadiums** is the more promising lead for genuine reuse:
  if a stadium's *broadcast camera convention* (which end is screen-left
  for the home broadcaster) were independently documented once, every
  match at that stadium *for the same broadcaster/competition* could
  reuse it — but no such database exists in this project or was created
  here (see §2's explicit finding that this would require new external
  research, not something derivable from the data itself).
- **International tournament matches** (Euro, World Cup, Women's Euro,
  Women's World Cup — 291 of 417 eligible matches, 70%) are concentrated
  in a small number of well-covered, high-profile events, which §2 found
  correlates with better public video availability (archive.org,
  official highlight channels) than domestic league matches.

---

## Part 2 — Public sources: what each can and cannot establish

Investigated via `WebSearch`/`WebFetch` (text/HTML only — no video
inspection). All findings below are **text-search results**, not
visually confirmed content.

### Structured data sources (checked explicitly, per the task's request)

- **The "Common Data Format" (CDF) for football match data**
  ([arXiv:2505.15820](https://arxiv.org/pdf/2505.15820)) proposes a
  `play_direction` field — but fetching the actual paper text confirmed
  this is defined by a **relative convention** ("CDF assumes that the
  home team *always* plays from left to right for the entire match"),
  explicitly **not** tied to a compass bearing or fixed stadium camera.
  It is also a **proposed schema specification, not a populated public
  dataset** — no real match values exist to query. This directly confirms
  the task's suspicion: even the broader football-data-standards
  community treats "attacking direction" as an *analytical* convention,
  identical in spirit to what this project's own `team_normalization.py`
  already does, not an *absolute physical* one.
- No other structured, queryable public database was found that directly
  records which physical goal a team defended in the first half of a
  specific match.

### Video/photographic sources (existence confirmed via search; content not inspected)

| Source type | Example found | Coverage pattern | Legitimacy/accessibility caveat |
|---|---|---|---|
| Internet Archive (archive.org) | User-uploaded Euro 2024/2020 full matches | Good for major tournaments, patchy for domestic leagues | User-uploaded, copyright status unclear |
| Official competition channels (UEFA, FIFA) | Match-centre "watch" pages, highlight clips | Reliable, official, but mostly **highlights**, not full matches with a clear early-match landmark | Access/rights vary by region |
| Fan-aggregator full-match sites (`fullmatchsports.cc`, `fullmatchsreplay.com`, `footballia.eu`) | Found candidates for La Liga, Ligue 1, Bundesliga matches | `footballia.eu` claims 51,000+ matches across 573 competitions, broadest single source found | Explicitly **not** rights-holder content — community-uploaded aggregators of uncertain legal standing; this report does not endorse or recommend using them, only notes they exist |
| Rights-holder streaming (MLS Season Pass / Apple TV) | Confirmed as the primary MLS source | Official, reliable | **Paywalled** — not freely accessible, a real barrier for this dataset's 3 MLS matches |

### Relative vs. absolute orientation — restated with new evidence

Every source investigated in Part 2 either (a) only records relative
attacking direction (CDF, and implicitly any text match report), or (b)
could in principle show absolute orientation but only via direct human
visual inspection of actual footage (video/photo sources) — there is
**no shortcut**. This matches and reinforces, rather than contradicts,
the prior investigation's core finding (`docs/physical_pitch_orientation_validation.md`):
StatsBomb data alone — and, now confirmed, no structured public football
dataset either — can establish absolute physical orientation. Only direct
visual evidence can.

### Broadcast camera placement from landmarks

Not independently investigated beyond what the Spain–Croatia case already
established (goalkeeper kit color, scoreboard, known goal side). No
general-purpose "stadium camera convention" database was found to exist
publicly. This remains an open research gap, not a solved problem — §8
recommends not pursuing it as a prerequisite.

---

## Part 3 — Manual calibration: effort evaluation

### Time breakdown, grounded in this session's actual measurements

| Step | Who | Measured/estimated time | Automatable? |
|---|---|---|---|
| Generate 2 candidate frames + review page | Assistant/script | **<1 second/match** (measured: 14 pages generated in a few seconds total) | Yes, fully — `scripts/generate_calibration_review_page.py` |
| Find candidate source link(s) | Assistant | **1 search query sufficed for a candidate in 14/14 pilot matches** (this session) | Yes, fully — demonstrated, could run unattended for all 417 |
| Open source, find the right moment, identify a landmark | **Human only** | Not measured (assistant cannot do this); estimated 3–15 min for an accessible, clear source | No |
| Compare landmark against the two candidate frames, decide | **Human only** | Estimated 1–3 min once the page is open | No |
| Record decision (fill and append the JSON snippet) | Human (or human + assistant pairing) | Estimated 1–2 min | Partially — the snippet is pre-generated; appending it could be scripted once approved |

### Aggregate estimate for 417 matches

- **Source discovery (assistant-automatable)**: ~1–2 minutes of tool time
  per match → roughly 7–14 hours of assistant tool-call time for all 417,
  schedulable as a background batch job. (Subject to the hosting
  platform's search-rate limits, not estimated here.)
- **Human verification (not automatable)**: optimistic 10 min/match → 70
  hours (~9 full workdays); realistic blended estimate 15–20 min/match →
  104–139 hours (~13–17 workdays); pessimistic (frequent dead ends) 25–30
  min/match → 174–209 hours (~22–26 workdays).
- **Expected proportion with accessible footage**: not reliably
  estimable from a 14-match sample; this pilot found *a* candidate link
  for 100% of matches but confirmed *accessible, watchable, full-match*
  video for a clearly smaller fraction (at best 1 direct match-specific
  video URL out of 14, `footballia.eu`'s Leverkusen–Darmstadt page — see
  §4). MLS (3 matches) is likely largely blocked by the Apple TV
  paywall. A reliable proportion figure requires actually attempting
  more matches, not extrapolating from 14.

### Is one verified frame per match sufficient?

Yes, for the renderer's actual, single binary decision (which team is
`--physical-reference-team`) — confirmed by this investigation's own
architecture: once period 1 is calibrated, `resolve_physical_reference_team_id`
(already implemented, tested) propagates correctly to every other period
via the Laws-of-the-Game end-swap rule (§5 below). One frame, carefully
chosen (ideally near a goal, for an unambiguous keeper/goal landmark —
exactly what `generate_calibration_review_page.py` already selects
automatically), is sufficient for the **horizontal** axis.

### Can the Y-axis be established too?

Partially, and with lower confidence than the X-axis, for the same reason
documented in the prior audit: the renderer has no separate Y-axis
control (the mirror formula moves X and Y together). A clear video
landmark (e.g. the penalty arc, corner flags, or a near/far touchline
pattern visible across multiple moments) *could* let a careful human
reviewer additionally judge depth/near-far consistency, but this requires
**continuous footage**, not one static frame — exactly why match
3930160's registry entry now distinguishes `horizontal_orientation_verified`
from `vertical_orientation_verified` as separate fields (§5), and why the
project owner's full-video comparison (not just the assistant's static
frame) was the stronger form of evidence that actually upgraded that
match's vertical-axis confidence.

### Preventing errors / independent validation

- Every calibration entry requires an explicit `evidence_landmark` and
  `source_url`/`source_description` (§5) — a second reviewer can always
  re-check the same source independently.
- `verification_status` has four distinct values (`verified`,
  `partially_verified`, `unverified`, `ambiguous`) specifically so a
  reviewer is never forced into a binary "trust it or discard it" choice
  — see §5.
- **Inter-reviewer agreement** is addressed in §8 (research-reproducibility
  section) rather than here, since it is a methodology question, not an
  engineering one.

---

## Part 4 — Representative pilot (14 matches, excluding Spain–Croatia)

### Sample selection

Stratified random sample (seed 42, one Python script, reproducible),
covering all 8 competitions, drawn from the real 417-match inventory —
**not hand-picked**:

| Match ID | Teams | Competition | Date |
|---|---|---|---|
| 3773403 | Barcelona vs Athletic Club | La Liga | 2021-01-31 |
| 3764661 | Athletic Club vs Barcelona | La Liga | 2021-01-06 |
| 3942226 | Spain vs Germany | UEFA Euro 2024 | 2024-07-05 |
| 3788776 | Sweden vs Poland | UEFA Euro 2020 | 2021-06-23 |
| 3802818 | PSG vs Lorient | Ligue 1 | 2022-04-03 |
| 3835333 | France Women's vs Belgium Women's | Women's Euro 2022 | 2022-07-14 |
| 3835327 | England Women's vs Norway Women's | Women's Euro 2022 | 2022-07-11 |
| 3857267 | Ecuador vs Senegal | FIFA World Cup 2022 | 2022-11-29 |
| 3857265 | Mexico vs Poland | FIFA World Cup 2022 | 2022-11-22 |
| 3877090 | LAFC vs Inter Miami | MLS | 2023-09-04 |
| 3901832 | Colombia Women's vs Jamaica Women's | Women's World Cup 2023 | 2023-08-08 |
| 3893791 | Spain Women's vs Costa Rica Women's | Women's World Cup 2023 | 2023-07-21 |
| 3895052 | Bayer Leverkusen vs RB Leipzig | 1. Bundesliga | 2023-08-19 |
| 3895067 | Bayer Leverkusen vs Darmstadt 98 | 1. Bundesliga | 2023-09-02 |

### Results (honest — zero new verifications)

| Match ID | Candidate source(s) found | Type | Verified? |
|---|---|---|---|
| 3773403 | ESPN match report; fullmatchsports.cc | Report / fan aggregator | **No — not opened** |
| 3764661 | ESPN match report; footballia.eu (general) | Report / broad archive, specific match not confirmed | **No** |
| 3942226 | Fox Sports highlights clip; archive.org search | Highlights / general archive search | **No** |
| 3788776 | ESPN match report; archive.org search | Report / general archive search | **No** |
| 3802818 | fullmatchsports.cc; fullmatchsreplay.com | Fan aggregators | **No** |
| 3835333 | ESPN match report | Report only | **No** |
| 3835327 | UEFA official report | Official report/highlights | **No** |
| 3857267 | NBC Sports highlights/recap | Highlights | **No** |
| 3857265 | **Official FIFA match-centre watch page** | Official, most promising lead | **No — not opened** |
| 3877090 | ESPN match page | Report; MLS likely paywalled for full match | **No** |
| 3901832 | Sky Sports report | Report only | **No** |
| 3893791 | ESPN match page | Report only | **No** |
| 3895052 | Bundesliga official match report | Official report | **No** |
| 3895067 | **footballia.eu direct match-specific page** | Most direct full-match-video lead found in the entire pilot | **No — not opened** |

**Pilot conclusion, stated exactly as required**: 14/14 matches yielded at
least one candidate source from a single search query each (confirming
source-discovery automation works well), but **0/14 were marked
verified** — the assistant cannot watch video, and no human review was
performed as part of this pilot. This is the correct, non-fabricated
result, not a shortfall: Part 4 explicitly asked to "document the
limitation, collect accessible source links, and prepare the review
material for human verification" when video inspection is unavailable —
done exactly as specified.

### Review material produced

`scripts/generate_calibration_review_page.py --match-id <id>` generates,
for any match: two real rendered candidate frames (team-A-reference vs.
team-B-reference, using the actual unmodified production renderer) plus
a metadata table, the candidate source links above (for the 14 pilot
matches; empty with a clear note for any other match), and a ready-to-fill
JSON snippet. Generated for all 14 pilot matches in
`results/physical_orientation/review_pages/<match_id>.html` (+ a
`<match_id>/` subfolder with the two PNGs). **These pages are ready for a
human to complete the verification step** — none have been marked
verified by this report.

---

## Part 5 — Calibration registry (implemented)

### Schema (`configs/physical_orientation_calibration.json`)

Implemented, following the exact "fail loudly on malformed input, never
silently treat a broken file as empty" pattern already established by
`configs/excluded_matches.json`/`src/match_exclusions.py`, with one
deliberate difference: a **missing** registry file returns an empty
mapping (not an error), since — unlike the exclusion manifest — this
registry is new and may legitimately not exist yet (see
`src/physical_orientation_calibration.py` docstring for the full
reasoning).

Each entry contains exactly the fields requested: `match_id`,
`period_1_physical_reference_team`, `period_handling_rule`,
`horizontal_orientation_verified`, `vertical_orientation_verified` (+
`vertical_orientation_note`), `source_url`, `source_description`,
`evidence_landmark`, `verification_method`, `confidence`,
`verification_status`, `verified_by`, `date_verified`, `reviewer_notes`,
`report`.

`verification_status` is a closed enum —
`verified` / `partially_verified` / `unverified` / `ambiguous` — enforced
by the loader (`CalibrationRegistryError` on any other value). **Only
`verified` entries are ever auto-applied** by
`resolve_usable_physical_reference_team_name()` — the other three statuses
are preserved and loadable (for review tooling) but never silently
trusted.

### One real entry: match 3930160

Already populated, reflecting the upgraded evidence from the project
owner's own full-video comparison of both halves (not just the two static
screenshots used in the prior session) — see
`configs/physical_orientation_calibration.json` for the complete entry.
**No other match has any entry** — nothing was fabricated to fill out the
registry.

### Period handling (already solved, reused here)

`period_1_physical_reference_team` captures exactly one fact: which team
attacks `+render-x` in period 1's arrangement. Propagation to periods 2–5
is handled automatically and was **not re-implemented** for this task —
it reuses `resolve_physical_reference_team_id()` and
`PHYSICALLY_FLIPPED_PERIODS = {2, 4}` exactly as built and tested in
`docs/full_physical_pitch_orientation_audit.md`. Extra time (periods 3–4)
and shootouts (period 5) are handled by the same existing, documented,
explicitly-flagged-as-an-assumption logic — no new work was needed or
done here.

---

## Part 6 — Automatic application in the renderer (implemented)

### Resolution priority (implemented in `src/animate_statsbomb_360_match.py`, `main()`)

1. **Explicit `--physical-reference-team`** (highest priority — always
   wins, exactly as the task requires: "Explicit CLI overrides remain
   available").
2. **A `verified` entry in the registry**, loaded automatically by
   `match_id` via `src/physical_orientation_calibration.py` — applied
   with the same period-aware mechanism as an explicit override.
3. **The existing automatic/analytical default** — unchanged, and now
   explicitly reports *why* it's being used ("No verified entry for this
   match... pass --physical-reference-team if you have evidence, or see
   docs/all_matches_physical_orientation_feasibility.md to calibrate it").

### Verified behavior (actually run, not just coded)

```
$ .venv/bin/python src/animate_statsbomb_360_match.py --match-id 3930160 ... (no --physical-reference-team)
Team normalization: reference_team_id=772 (period-aware physical calibration, auto-applied from
configs/physical_orientation_calibration.json (verified entry, team='Spain'; ...)
(per-period reference_team_id: {1: 772, 2: 785}; ...) REFERENCE=1574 MIRRORED=1679

$ .venv/bin/python src/animate_statsbomb_360_match.py --match-id 3857283 ... (uncalibrated match)
Team normalization: reference_team_id=782 (automatic (chronologically-first team; NOT physically
meaningful -- see docs/physical_pitch_orientation_validation.md). No verified entry for this match...)
```

Both confirmed by direct execution in this session. **Physical
orientation is never hidden**: every run's own stdout states explicitly
whether the result is CLI-overridden, registry-calibrated, or
uncalibrated/analytical — satisfying "Never hide missing physical
calibration."

### Preserved flags and behavior (confirmed unchanged)

`--apply-orientation-correction`, `--apply-team-normalization`,
`--apply-team-identity-resolution`, `--view camera-overlay`,
`--timing-mode real-time` — all present, unmodified, and used exactly as
before in every command in this report. Raw StatsBomb data and
`src/build_statsbomb_360_dataset.py` (the ML dataset builder) were not
touched.

---

## Part 7 — Human review workflow (implemented, lightweight, static)

### Design decision: static HTML, not an application

Evaluated and rejected a server/web-app approach, per the task's own
instruction ("Do not implement a large application before evaluating
whether a simpler workflow is sufficient"). A single-user, local,
bachelor-thesis research tool does not need a backend: a static HTML
file per match, generated by a short script, opened directly in a
browser, is sufficient and was built
(`scripts/generate_calibration_review_page.py`).

### Why two candidates, not four

Directly addressed, with reasoning: the actual renderer exposes exactly
one binary physical-calibration choice (`--physical-reference-team` is
either team A or team B). A third/fourth "horizontal-only"/"vertical-only"
variant is **not a real, selectable pipeline option** — those were used
only as one-off diagnostic renders in the prior investigation
(`docs/full_physical_pitch_orientation_audit.md`), not exposed via any
CLI flag. Showing a reviewer a choice they cannot actually make in the
real tool would be actively misleading. The review page therefore shows
exactly the two real candidates, both rendered by the real, unmodified
production function (`render_event_frame`).

### What the page contains (confirmed, all 14 pilot pages generated and inspected)

Match metadata, the representative event (automatically chosen: the
first `Shot` in the requested period if one exists — otherwise the
temporal midpoint — so the reviewer gets a goal-adjacent, landmark-rich
frame whenever possible, exactly as demonstrated for match 3773403's
Messi shot), the two candidate frame images, the candidate source links
researched for that match, and a pre-filled JSON snippet ready to
complete and append to the registry.

### What the page does NOT do

It never writes to the registry itself — recording a calibration is
always a separate, explicit, human-performed file edit. This was a
deliberate choice to guarantee no calibration can ever be silently
auto-recorded by a script.

---

## Part 8 — Research reproducibility (thesis methodology guidance)

### Reproducibility

Every calibration decision is traceable to: (a) a specific source URL or
description, (b) a specific landmark description, (c) a verifier
identity, (d) a date. A second person can, in principle, re-open the same
source and re-check the same landmark — this is the standard for
reproducibility available (full re-verification by an independent third
party was not performed in this session).

### Evidence traceability

The registry schema (§5) was designed specifically so no field is ever
inferred silently — `source_url`/`source_description` and
`evidence_landmark` are both required, non-optional-in-spirit fields
(enforced as required by the loader's `_REQUIRED_FIELDS`... note: of the
required set, `evidence_landmark` is required; `source_url` is optional
since some evidence — like the project owner's own direct video
comparison — is not URL-addressable, but `source_description` then
carries the burden of describing it, as done for match 3930160).

### Human verification and potential annotation errors

Recommend documenting, for the thesis: (1) who performed each
verification, (2) what specific landmark was used, (3) that the
assistant's role was limited to candidate discovery and rendering, never
independent visual verification, for any match without directly-supplied
screenshots. This is already the case for match 3930160's registry entry
and should remain the pattern.

### Inter-reviewer agreement

Not measured in this session (only one verified match, one reviewer).
**Recommendation for the thesis methodology section**: if time permits,
have a second person independently verify a small subset (e.g. 5–10
matches) already marked `verified` by the primary reviewer, and report
simple percent agreement — this is a standard, cheap way to report
annotation reliability without requiring every match to be double-checked.

### Does physical calibration affect ML features, or only presentation?

**Only presentation/visualization.** This is a direct, confirmed
consequence of the architecture: `src/build_statsbomb_360_dataset.py`
(the ML pipeline) calls `resolve_reference_team_id(ordered_events)` with
a single positional argument — it never passes
`physical_reference_team_name`, never reads the calibration registry, and
was not modified in this or the prior session. The ML dataset's
analytical coordinate convention is completely independent of, and
unaffected by, any physical calibration work. **This should be stated
explicitly in the thesis's data-preprocessing section** to avoid a reader
assuming physical calibration changes model inputs — it does not.

### Recommended methodology-section framing

"Physical pitch-orientation calibration was performed for visualization
purposes only, independently of the analytical coordinate system used
for machine-learning features. A calibration was considered verified
only when an independent, citable source showed an unambiguous landmark
(e.g. a goalkeeper's identifiable kit color at a specific goal) matching
the StatsBomb-derived rendering at the same match timestamp. All
calibration decisions, their evidence, and their confidence level are
recorded in `configs/physical_orientation_calibration.json` and are
independently re-checkable."

---

## Part 9 — Recommendation

**Hybrid approach**, not pure-manual and not pure-automatic:

1. **Automation (assistant-performed)**: source-candidate discovery via
   `WebSearch`, and candidate-frame/review-page generation via
   `scripts/generate_calibration_review_page.py`. Both are implemented,
   tested, and demonstrated at pilot scale (14 matches, seconds of
   compute). Could be run for all 417 matches as an unattended batch
   without further engineering work.
2. **Manual (human-performed, cannot be automated)**: opening each
   candidate source, identifying a landmark, comparing against the two
   rendered candidates, and recording the decision. Estimated at
   15–20 minutes/match on a realistic blended basis (§3).
3. **Start with a validated subset**, not all 417 at once — per the
   task's own explicit instruction not to launch an uncontrolled 417-match
   review before presenting findings (done: this report *is* that
   presentation). Recommend the next concrete step be a further 20–30
   match batch, prioritized by which matches actually appear in planned
   thesis figures/analysis (not an arbitrary continuation of the random
   pilot sample).

---

## Deliverables produced in this session

| File | Purpose |
|---|---|
| `scripts/build_match_orientation_inventory.py` | Reproducible inventory builder (Part 1) |
| `results/physical_orientation/match_inventory.csv` | Full 426-row inventory |
| `src/physical_orientation_calibration.py` | Registry loader (Part 5), 15 new tests |
| `configs/physical_orientation_calibration.json` | Registry with the 1 verified entry |
| `src/animate_statsbomb_360_match.py` (modified) | Auto-load wiring (Part 6) |
| `scripts/generate_calibration_review_page.py` | Review-page generator (Part 7) |
| `results/physical_orientation/review_pages/*.html` (+ 14 subfolders of PNGs) | Pilot review material (Part 4) |
| `docs/all_matches_physical_orientation_feasibility.md` | This report |

**Full test suite: 172 passed** (157 from the prior session + 15 new
registry tests), zero regressions. Nothing was committed or staged.

## Remaining uncertainties (explicit, not glossed over)

1. No match beyond 3930160 has been visually verified against any
   footage by anyone in this session — the pilot's 14 matches remain
   fully unverified, by design.
2. The "expected proportion of matches with accessible footage" cannot be
   reliably estimated from a 14-match sample; only a larger attempt (or
   the full 417) would give a trustworthy figure.
3. No stadium-camera-convention database was built or found; reuse across
   matches sharing a stadium remains theoretical, not implemented.
4. Fan-aggregator video sources found in Part 2/4 carry unresolved
   copyright/reliability questions this report does not attempt to
   adjudicate — noted, not endorsed.
