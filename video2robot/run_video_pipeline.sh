set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <video_path> [output_root]" >&2
  exit 1
fi

VIDEO_PATH=$(realpath "$1")
OUTPUT_ROOT=${2:-video2real_outputs}
MEGASAM_ROOT=${MEGASAM_ROOT:-mega-sam}
SCENE_ARG=${SCENE:-}
REUSE_MEGASAM=${REUSE_MEGASAM:-1}
SKIP_FRAMES=${SKIP_FRAMES:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "[run_video_pipeline] ERROR: video file not found: $VIDEO_PATH" >&2
  exit 1
fi

VIDEO_NAME=$(basename "${VIDEO_PATH%.*}")
OUTPUT_DIR=$(realpath "$OUTPUT_ROOT")/"$VIDEO_NAME"
mkdir -p "$OUTPUT_DIR"

PIPELINE_ARGS=("$VIDEO_PATH" "--megasam-root" "$MEGASAM_ROOT" "--output-dir" "$OUTPUT_DIR")
if [[ -n "$SCENE_ARG" ]]; then
  PIPELINE_ARGS+=("--scene" "$SCENE_ARG")
fi
if [[ "$REUSE_MEGASAM" == "0" ]]; then
  PIPELINE_ARGS+=("--no-reuse")
fi

echo "[run_video_pipeline] Starting pipeline for $VIDEO_PATH"
echo "[run_video_pipeline] Output directory: $OUTPUT_DIR"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[run_video_pipeline] DRY_RUN=1 set; skipping python execution"
else
  python3 -m video2real.run_pipeline "${PIPELINE_ARGS[@]}"
fi

echo "[run_video_pipeline] Normalizing output filenames"
if [[ -f "$OUTPUT_DIR/aligned.json" ]]; then
  mv -f "$OUTPUT_DIR/aligned.json" "$OUTPUT_DIR/align.json"
fi
if [[ -f "$OUTPUT_DIR/wrist_g1.json" ]]; then
  mv -f "$OUTPUT_DIR/wrist_g1.json" "$OUTPUT_DIR/g1.json"
fi

echo "[run_video_pipeline] Done. Contents of $OUTPUT_DIR:"
ls "$OUTPUT_DIR"
