# Team-identity manual review package (2026-10-24)

Status: **prepared for your visual review, not validated.** Nothing was recoloured or corrected. The
colours are exactly what the production renderer draws, including the display-only shared-snapshot
refinement already in place.

Reproduce: `.venv/bin/python scripts/build_team_identity_manual_review.py --seed 20261024`
(selection details in `selection_log.txt`).

## Selection

- **Source:** the 102 strong frames of `team_identity_investigation/colour_inconsistent_frames.csv`
  (colour contradicts both neighbouring snapshots). Frame 505 is excluded explicitly; it is not in
  the strong set anyway, since it has only one neighbour.
- **Strata** (sizes in brackets):
  - full inversion, ≥ 90% of paired players contradict both neighbours (6): 2 cases;
  - mixed, exactly 4 contradictions against the weaker neighbour (66): 1 case;
  - mixed, 5–7 (22): 1 case;
  - mixed, ≥ 8 (8): 1 case.
- **Within each stratum:** a seeded shuffle; the first candidate with an unused match and an unused
  competition is taken.
- **Result:** 5 matches, 5 competitions, periods 1 and 2. No case was hand-picked.
- **Count:** you asked for 5 cases, so there are `case_01`–`case_05` (the tree in the request showed
  up to `case_10`).

## Per case

- `context.gif`: production CLI with `--apply-orientation-correction --apply-team-normalization
  --apply-team-identity-resolution --view camera-overlay --timing-mode real-time`. It runs from 8 s
  before the previous neighbour to 8 s after the next. Real snapshots only: no interpolation, and gaps
  are shown with their real duration.
- `comparison.png`:
  - top-left: PREVIOUS snapshot;
  - top-right: SUSPICIOUS as drawn;
  - bottom-left: NEXT snapshot;
  - bottom-right: SUSPICIOUS with rings, **white** = this player's colour contradicts the paired player
    in BOTH neighbours, **orange** = contradicts one.
- `points.csv`: per point, the raw `teammate` flag, the drawn team, the paired team before and after,
  and the pairing distances. Players are paired by optimal assignment; pairs over 3 yd are not judged.
- `context_frames.csv`: the GIF's frame list (the suspicious frame number is in `review_index.csv`).

visible_area is present in every frame of every GIF, and every frame has players (minimum 7).

## What to look for

| Case | Match (competition) | Suspicious frame | Pattern | Look for | Uncertainty |
|---|---|---|---|---|---|
| **case_01** | 3893791 Spain W. – Costa Rica W. (Women's World Cup) | P1 18:25.761, Costa Rica **Duel**, GIF frame 3 of 4 | **Full inversion:** all 18 players contradict both neighbours | Frames 2, 3 and 4 are three events at the **same instant** (Spain Ball Receipt*, Costa Rica Duel, Spain Shot) with almost the same positions. Every dot changes colour in frame 3 and changes back in frame 4. | The neighbours are separate snapshots of the same moment, and they agree with each other. Problem A rotated this snapshot, so it was recorded in Spain's frame, and its `teammate` flags are likely Spain's view, never recomputed for the Costa Rica event. This is a hypothesis. Only 4 snapshots exist in the ±8 s window (data). |
| **case_02** | 3895107 FC Köln – Bayer Leverkusen (Bundesliga) | P2 82:13.060, Köln **Block**, GIF frame 17 | **Full inversion**, 14 both + 2 one-sided | Frames 16 (Leverkusen Block, 0.08 s earlier), 17 and 18 (Leverkusen Pressure): the same positions, colours swapped only in frame 17 | Two blocks, one per team, 0.08 s apart. The suspicious snapshot looks like the other team's view (same pattern as case_01). It is not byte-identical to a neighbour, so the shared-snapshot logic cannot see it. |
| **case_03** | 3857256 Switzerland – Serbia (FIFA World Cup) | P2 90:23.300, Serbia **Pass**, GIF frame 9 | **Mixed, weak:** 2 both + 6 one-sided | The packed cluster at the left goal-line between frames 8, 9 and 10. Do individual dots change colour, or do players just stand very close together? | **Low confidence.** In a goal-mouth crowd, pairing players between frames can mix up neighbours. Only 2 players contradict both neighbours. It may be a minor raw-flag error or a pairing artefact. |
| **case_04** | 3844386 Sweden W. – Belgium W. (UEFA Women's Euro) | P1 31:58.616, Sweden **Pressure**, GIF frame 3 | **Mixed, weak:** 2 both + 6 one-sided | The crowd around Belgium's goal (keeper diamond) across frames 2, 3 and 4 | **Low confidence**, for the same reason as case_03. The keeper is coloured consistently. |
| **case_05** | 3773386 Barcelona – Deportivo Alavés (La Liga) | P1 27:13.282, Barcelona **Ball Receipt*** (Jordi Alba), GIF frame 9 | **Mixed, strong:** 10 both + 5 one-sided | Frame 8 (Barcelona Pass) → 9 → 10 (Alavés Block at the same instant). In frame 9 the actor ring sits in Alavés's goal, where frame 8 shows Alavés's keeper, and it is drawn in Barcelona colour; many dots in the box swap. | A failed receipt together with the other team's same-moment Block: the case_01/02 pattern, but only partly inverted. Problem A is UNCERTAIN here; the orientation came from the display rules. Which reading is right for each player is not determinable from the data. |

## Observations for your judgement (not conclusions)

1. Cases 01, 02 and 05 share a pattern. A snapshot belonging to one team's event at a given moment
   appears with the other team's `teammate` view on a same-moment event of the other team (Duel, Block,
   failed Ball Receipt*). These look like near-duplicates of a snapshot rather than byte-identical
   copies, so the existing shared-snapshot detection (Problem C) does not apply.
2. Cases 03 and 04 may not be identity errors at all. Your visual check decides whether the colour
   changes are real.
3. Possible follow-ups, only after your observations:
   - extend Problem C's idea to near-duplicate same-instant snapshots (display-only, like the earlier
     refinement);
   - or draw such frames' points in the neutral "team unclear" colour;
   - or keep them as an acknowledged data limitation.

`review_index.csv` has, per case: match ID, competition, teams, period, clock, timestamp, event ID and
type, event team, shared flag, Problem A result, number of suspicious players (vs both and vs one
neighbour), the audit's counts, the neighbour clocks, the stratum and selection reason, and the GIF
frame of the suspicious snapshot.
