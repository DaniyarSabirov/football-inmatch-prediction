# Team-identity investigation (2026-10-24)

Triggered by the reviewer's note on sheet `uncertain_frames/3930164_p1_000000_505.png`.

## Frame 505: Slovakia vs Belgium (3930164), P1 00:00.505, Kick Off pass (Lukáš Haraslín)

Event id `5aabea4a-a1e7-4e5e-adc3-b13dec0f27bf`, raw event index 5 (the first event after Starting XI
and Half Start), play pattern "From Kick Off". It has its own 360 snapshot with 20 points and no
duplicate: it is not shared with any other event, so Problem C (shared-snapshot identity) does not
apply. The renderer colours each point straight from StatsBomb's raw `teammate` flag, relative to
the event's team (Slovakia).

Findings (`frame_505/points.csv`, `frame_505/sheet.png`, `frame_505/opening_10s.gif`):

| Check | Result |
|---|---|
| Orientation | Correct as drawn: the points lie 1.94 yd on average from their positions in the next snapshot (00:01.616), against 6.77 yd if rotated. This matches the reviewer's CURRENT choice. |
| Ball in play? | No. The frame is the kick-off itself, 0.5 s into the period, so every player must be in their own half (Law 8). No one can have moved 20+ yards. |
| Law 8 by drawn colour | 8 of 20 points are coloured as a team standing in the opponent's half (6 "Slovakia" at x = 68–84, 2 "Belgium" at x = 32–36); 7 are on the halfway line. |
| Continuity (optimal pairing with the next snapshot) | 11 of 20 points change colour. Swapping all flags does not help (8 vs 12 either way), so this is a scramble, not a global inversion. |
| Next snapshots (00:01.616, 00:02.446) | Consistent: Slovakia in its own half and on the halfway line, Belgium in its own half. |
| Rest of the match | Only weak single-neighbour flags otherwise (4/5, 4/7, two of 4/11); the opening minutes after 505 are clean. |

**Conclusion: confirmed team-colour error in this frame, caused by the raw StatsBomb 360 `teammate`
flags.** It is not caused by orientation, team normalization, Problem C or colour assignment. There is
no per-player identity in the data that would allow a reliable recolouring, so nothing was recoloured.
The frame review records the CURRENT orientation and keeps the colour question as an `open_issue`.

## Dataset-wide (`scripts/audit_team_identity_continuity.py`)

Consecutive distinct snapshots (same period, ≤ 2 s apart) were paired by optimal assignment, keeping
pairs ≤ 3 yd apart. A frame is flagged when ≥ 4 paired players and ≥ 30% of pairs change colour
against every available neighbour.

| | Before refinement | After |
|---|---|---|
| Paired player observations | 5,779,934 | 5,779,934 |
| … changing colour (includes pairing mistakes in crowds) | 283,029 | 282,979 |
| Flagged frames, both neighbours (strong) | 104 | 102 |
| Flagged frames, single neighbour (weak: one bad link flags both of its frames) | 2,095 | 2,094 |
| Kick-off frames flagged | 1 (= 505) | 1 |

Strong cases: 8 are full inversions and 96 partial scrambles. There is no systematic pattern
(event team vs possession team mixed), so these are raw-data errors.

**Pipeline error found and fixed (display only):** in 2 shared snapshots in Uruguay vs Ghana (3857293:
P1 44:24.954 and P2 83:27.375, a failed Ball Receipt* sharing its snapshot with Ghana's Block), Problem C
read the `teammate` flags via the wrong event. Colours contradicted both neighbours for every paired
player (12/12, 13/13 and 17/17, 8/8); the other event's reading contradicts none. Ghana's goalkeeper, in
Ghana's own goal, was drawn in Uruguay's colour. See `shared_source_3857293/*.png`.

`refine_shared_sources_by_continuity` (renderer, display only) switches the source only for shared
snapshots, only with non-shared neighbours on both sides within 2 s, only when each neighbour disagrees
on ≥ 4 players and ≥ 50% of pairs, and only when the alternative disagrees on ≤ 10%. It switched
exactly these 2 groups. Two plausible cases stay unswitched under the strict rule: 3893827 P1 45:54.120
(8/9 vs 1/9) and 3835321 P1 32:09.617 (4/5 vs 1/5).

`src/team_identity.py` is not changed. It is also used by `build_statsbomb_360_dataset.py`, so the ML
features of these 2 snapshots keep Problem C's reading. That is a separate decision for you.

## Not fixed (data limitation, by design)

- Frame 505 and the other raw `teammate` scrambles: no reliable per-player correction exists.
- Possible policy (not implemented, needs your decision): draw the points of strongly flagged frames
  in the neutral "team unclear" colour instead of their raw colours.
