#!/bin/bash

set -euo pipefail

MATCH_ID=3902240
PERIOD=1
START_TIME="10:00"
END_TIME="10:30"
THEME="dark"
VIEW="analysis"
TIMING_MODE="real-time"
PLAYBACK_SPEED="1.0"
FRAME_DURATION_MS=250
MINIMUM_FRAME_DURATION_MS=80
MAXIMUM_FRAME_DURATION_MS=1500
MAXIMUM_RENDER_FRAMES=0
ALL_EVENTS=false
ALL_PERIOD_EVENTS=false
SHOW_TIMING_DEBUG=false
HIGHLIGHT_GOALKEEPERS=false

usage() {
  cat <<'EOF'
Usage:
  scripts/run_360_preview.sh [OPTIONS]
  scripts/run_360_preview.sh START_SECOND END_SECOND [THEME]  # legacy

Selection options:
  --match-id ID
  --period PERIOD
  --start MM:SS|HH:MM:SS
  --end MM:SS|HH:MM:SS
  --all-period-events
  --all-events

Display and safety options:
  --theme light|dark
  --view analysis|camera-overlay|full-pitch|visible-area
  --timing-mode fixed|real-time
  --playback-speed SPEED
  --frame-duration-ms MS
  --minimum-frame-duration-ms MS
  --maximum-frame-duration-ms MS
  --maximum-render-frames COUNT
  --show-timing-debug
  --highlight-goalkeepers

Examples:
  scripts/run_360_preview.sh --match-id 3902240 --period 1 --start 20:00 --end 30:00 --theme dark
  scripts/run_360_preview.sh --match-id 3902240 --period 1 --all-period-events --theme dark
  scripts/run_360_preview.sh --match-id 3902240 --all-events --theme dark
EOF
}

is_non_negative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

parse_time_seconds() {
  local value="$1"
  local first second third
  IFS=: read -r first second third <<< "$value"
  if [[ -z "${first}" || -z "${second}" ]] || ! is_non_negative_integer "$first" || ! is_non_negative_integer "$second"; then
    return 1
  fi
  if [[ -n "${third:-}" ]]; then
    if ! is_non_negative_integer "$third" || (( second >= 60 || third >= 60 )); then
      return 1
    fi
    echo $((10#$first * 3600 + 10#$second * 60 + 10#$third))
  else
    if (( second >= 60 )); then
      return 1
    fi
    echo $((10#$first * 60 + 10#$second))
  fi
}

# Preserve the old positional START_SECOND END_SECOND [THEME] shortcut.
if (( $# > 0 )) && [[ "$1" != --* ]]; then
  if (( $# > 3 )); then
    echo "Error: legacy positional usage accepts at most three arguments."
    usage
    exit 1
  fi
  if ! is_non_negative_integer "$1" || [[ -z "${2:-}" ]] || ! is_non_negative_integer "$2"; then
    echo "Error: positional start and end must be non-negative integers."
    usage
    exit 1
  fi
  if (( 10#$2 <= 10#$1 )); then
    echo "Error: end second must be greater than start second."
    usage
    exit 1
  fi
  START_TIME="$((10#$1 / 60)):$(printf '%02d' $((10#$1 % 60)))"
  END_TIME="$((10#$2 / 60)):$(printf '%02d' $((10#$2 % 60)))"
  THEME="${3:-dark}"
  shift "$#"
fi

while (( $# > 0 )); do
  case "$1" in
    --match-id) MATCH_ID="${2:?Missing value for --match-id}"; shift 2 ;;
    --period) PERIOD="${2:?Missing value for --period}"; shift 2 ;;
    --start) START_TIME="${2:?Missing value for --start}"; shift 2 ;;
    --end) END_TIME="${2:?Missing value for --end}"; shift 2 ;;
    --all-period-events) ALL_PERIOD_EVENTS=true; shift ;;
    --all-events) ALL_EVENTS=true; shift ;;
    --theme) THEME="${2:?Missing value for --theme}"; shift 2 ;;
    --view) VIEW="${2:?Missing value for --view}"; shift 2 ;;
    --timing-mode) TIMING_MODE="${2:?Missing value for --timing-mode}"; shift 2 ;;
    --playback-speed) PLAYBACK_SPEED="${2:?Missing value for --playback-speed}"; shift 2 ;;
    --frame-duration-ms) FRAME_DURATION_MS="${2:?Missing value for --frame-duration-ms}"; shift 2 ;;
    --minimum-frame-duration-ms) MINIMUM_FRAME_DURATION_MS="${2:?Missing value for --minimum-frame-duration-ms}"; shift 2 ;;
    --maximum-frame-duration-ms) MAXIMUM_FRAME_DURATION_MS="${2:?Missing value for --maximum-frame-duration-ms}"; shift 2 ;;
    --maximum-render-frames) MAXIMUM_RENDER_FRAMES="${2:?Missing value for --maximum-render-frames}"; shift 2 ;;
    --show-timing-debug) SHOW_TIMING_DEBUG=true; shift ;;
    --highlight-goalkeepers) HIGHLIGHT_GOALKEEPERS=true; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Error: unknown argument: $1"; usage; exit 1 ;;
  esac
done

if ! is_non_negative_integer "$MATCH_ID" || ! is_non_negative_integer "$PERIOD" || (( PERIOD < 1 )); then
  echo "Error: match ID must be non-negative and period must be positive."
  exit 1
fi
if [[ "$THEME" != "dark" && "$THEME" != "light" ]]; then
  echo "Error: theme must be dark or light."
  exit 1
fi
if [[ "$VIEW" != "analysis" && "$VIEW" != "camera-overlay" && "$VIEW" != "full-pitch" && "$VIEW" != "visible-area" ]]; then
  echo "Error: unsupported view: $VIEW"
  exit 1
fi
if [[ "$TIMING_MODE" != "fixed" && "$TIMING_MODE" != "real-time" ]]; then
  echo "Error: timing mode must be fixed or real-time."
  exit 1
fi
for value in "$FRAME_DURATION_MS" "$MINIMUM_FRAME_DURATION_MS" "$MAXIMUM_FRAME_DURATION_MS" "$MAXIMUM_RENDER_FRAMES"; do
  if ! is_non_negative_integer "$value"; then
    echo "Error: frame duration and limit values must be non-negative integers."
    exit 1
  fi
done
if [[ "$ALL_EVENTS" == true && "$ALL_PERIOD_EVENTS" == true ]]; then
  echo "Error: --all-events and --all-period-events are incompatible."
  exit 1
fi

TIMING_SLUG="${TIMING_MODE//-/_}"
PYTHON_ARGS=(
  src/animate_statsbomb_360_match.py
  --match-id "$MATCH_ID"
  --theme "$THEME"
  --view "$VIEW"
  --timing-mode "$TIMING_MODE"
  --playback-speed "$PLAYBACK_SPEED"
  --frame-duration-ms "$FRAME_DURATION_MS"
  --minimum-frame-duration-ms "$MINIMUM_FRAME_DURATION_MS"
  --maximum-frame-duration-ms "$MAXIMUM_FRAME_DURATION_MS"
  --maximum-render-frames "$MAXIMUM_RENDER_FRAMES"
)

if [[ "$ALL_EVENTS" == true ]]; then
  SELECTION_SLUG="all_periods_all_events"
  SELECTION_DESCRIPTION="all periods / all usable events"
  PYTHON_ARGS+=(--all-events)
elif [[ "$ALL_PERIOD_EVENTS" == true ]]; then
  SELECTION_SLUG="period_${PERIOD}_all_events"
  SELECTION_DESCRIPTION="period ${PERIOD} / all usable events"
  PYTHON_ARGS+=(--period "$PERIOD" --all-period-events)
else
  START_SECONDS="$(parse_time_seconds "$START_TIME")" || { echo "Error: invalid --start time: $START_TIME"; exit 1; }
  END_SECONDS="$(parse_time_seconds "$END_TIME")" || { echo "Error: invalid --end time: $END_TIME"; exit 1; }
  if (( END_SECONDS <= START_SECONDS )); then
    echo "Error: end time must be greater than start time."
    exit 1
  fi
  SELECTION_SLUG="period_${PERIOD}_seconds_${START_SECONDS}_${END_SECONDS}"
  SELECTION_DESCRIPTION="period ${PERIOD}, [${START_TIME}, ${END_TIME})"
  PYTHON_ARGS+=(--period "$PERIOD" --start-time "$START_TIME" --end-time "$END_TIME")
fi

OUTPUT_PREFIX="results/statsbomb_360_match_${MATCH_ID}_${SELECTION_SLUG}_${THEME}_${TIMING_SLUG}"
GIF_PATH="${OUTPUT_PREFIX}.gif"
MANIFEST_PATH="${OUTPUT_PREFIX}_event_frames.csv"
TIMING_PLOT_PATH="${OUTPUT_PREFIX}_timing.png"
PYTHON_ARGS+=(--output "$GIF_PATH" --event-frame-manifest "$MANIFEST_PATH")
if [[ "$ALL_EVENTS" != true ]]; then
  PYTHON_ARGS+=(--timing-plot "$TIMING_PLOT_PATH")
fi
if [[ "$SHOW_TIMING_DEBUG" == true ]]; then PYTHON_ARGS+=(--show-timing-debug); fi
if [[ "$HIGHLIGHT_GOALKEEPERS" == true ]]; then PYTHON_ARGS+=(--highlight-goalkeepers); fi

echo "Running generic StatsBomb 360 event preview"
echo "Match ID: ${MATCH_ID}"
echo "Selection: ${SELECTION_DESCRIPTION}"
echo "Theme: ${THEME}"
echo "View: ${VIEW}"
echo "Timing mode: ${TIMING_MODE}"
echo "Maximum render frames: ${MAXIMUM_RENDER_FRAMES}"
echo "GIF: ${GIF_PATH}"
echo "Event-frame manifest: ${MANIFEST_PATH}"
if [[ "$ALL_EVENTS" != true ]]; then echo "Timing plot: ${TIMING_PLOT_PATH}"; fi

python "${PYTHON_ARGS[@]}"

echo "Preview generated successfully."
open "$GIF_PATH"
