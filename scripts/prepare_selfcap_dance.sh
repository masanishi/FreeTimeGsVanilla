#!/bin/bash
set -euo pipefail

VIDEO_DIR="${VIDEO_DIR:-$(pwd)/dance}"
DATA_DIR="${DATA_DIR:-$(pwd)/dataset/selfcap_dance}"
NUM_CAMERAS="${NUM_CAMERAS:-24}"
NUM_FRAMES="${NUM_FRAMES:-60}"
RESIZE_SCALE="${RESIZE_SCALE:-0.5}"
KEYFRAME_STEP="${KEYFRAME_STEP:-5}"
ROMA_REF_CAM="${ROMA_REF_CAM:-0000}"
ROMA_DEVICE="${ROMA_DEVICE:-auto}"
ROMA_CERTAINTY="${ROMA_CERTAINTY:-0.3}"
ROMA_MIN_DEPTH="${ROMA_MIN_DEPTH:-1e-4}"
ROMA_IMAGE_SCALE="${ROMA_IMAGE_SCALE:-1.0}"
ROMA_AMP="${ROMA_AMP:-0}"
ROMA_USE_RANSAC="${ROMA_USE_RANSAC:-0}"
ROMA_RANSAC_TH="${ROMA_RANSAC_TH:-0.5}"
ROMA_VOXEL_SIZE="${ROMA_VOXEL_SIZE:-0}"
ROMA_PER_FRAME="${ROMA_PER_FRAME:-1}"
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"
SKIP_COLMAP="${SKIP_COLMAP:-0}"
SKIP_ROMA="${SKIP_ROMA:-0}"
SKIP_COMBINE="${SKIP_COMBINE:-0}"

usage() {
  echo "Usage: $0 [--video-dir PATH] [--data-dir PATH] [--num-cameras N] [--num-frames N]"
  echo "          [--resize-scale S] [--keyframe-step N]"
  echo "          [--roma-ref-cam ID] [--roma-device DEV] [--roma-certainty TH]"
  echo "          [--roma-min-depth D] [--roma-image-scale S] [--roma-amp]"
  echo "          [--roma-use-ransac] [--roma-ransac-th PX] [--roma-voxel-size M]"
  echo "          [--roma-per-frame 0|1]"
  echo "          [--skip-extract] [--skip-colmap] [--skip-roma] [--skip-combine]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir)
      VIDEO_DIR="$2"; shift 2 ;;
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;
    --num-cameras)
      NUM_CAMERAS="$2"; shift 2 ;;
    --num-frames)
      NUM_FRAMES="$2"; shift 2 ;;
    --resize-scale)
      RESIZE_SCALE="$2"; shift 2 ;;
    --keyframe-step)
      KEYFRAME_STEP="$2"; shift 2 ;;
    --roma-ref-cam)
      ROMA_REF_CAM="$2"; shift 2 ;;
    --roma-device)
      ROMA_DEVICE="$2"; shift 2 ;;
    --roma-certainty)
      ROMA_CERTAINTY="$2"; shift 2 ;;
    --roma-min-depth)
      ROMA_MIN_DEPTH="$2"; shift 2 ;;
    --roma-image-scale)
      ROMA_IMAGE_SCALE="$2"; shift 2 ;;
    --roma-amp)
      ROMA_AMP=1; shift ;;
    --roma-use-ransac)
      ROMA_USE_RANSAC=1; shift ;;
    --roma-ransac-th)
      ROMA_RANSAC_TH="$2"; shift 2 ;;
    --roma-voxel-size)
      ROMA_VOXEL_SIZE="$2"; shift 2 ;;
    --roma-per-frame)
      ROMA_PER_FRAME="$2"; shift 2 ;;
    --skip-extract)
      SKIP_EXTRACT=1; shift ;;
    --skip-colmap)
      SKIP_COLMAP=1; shift ;;
    --skip-roma)
      SKIP_ROMA=1; shift ;;
    --skip-combine)
      SKIP_COMBINE=1; shift ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

IMAGES_DIR="$DATA_DIR/images"
COLMAP_DIR="$DATA_DIR/colmap"
TRIANGULATION_DIR="$DATA_DIR/triangulation"
NPZ_PATH="$DATA_DIR/keyframes_${NUM_FRAMES}frames_step${KEYFRAME_STEP}.npz"

ROMA_ARGS=(
  --images-dir "$IMAGES_DIR"
  --colmap-model "$COLMAP_DIR/sparse/0"
  --output-dir "$TRIANGULATION_DIR"
  --frame-step 1
  --ref-cam "$ROMA_REF_CAM"
  --device "$ROMA_DEVICE"
  --certainty "$ROMA_CERTAINTY"
  --min-depth "$ROMA_MIN_DEPTH"
  --image-scale "$ROMA_IMAGE_SCALE"
  --voxel-size "$ROMA_VOXEL_SIZE"
)

if [[ $ROMA_AMP -eq 1 ]]; then
  ROMA_ARGS+=(--amp)
fi
if [[ $ROMA_USE_RANSAC -eq 1 ]]; then
  ROMA_ARGS+=(--use-ransac --ransac-th "$ROMA_RANSAC_TH")
fi

mkdir -p "$DATA_DIR"

if [[ $SKIP_EXTRACT -eq 0 ]]; then
  echo "==> Extracting frames (run inside tmux recommended)"
  bash scripts/extract_selfcap_frames.sh \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$IMAGES_DIR" \
    --num-cameras "$NUM_CAMERAS" \
    --num-frames "$NUM_FRAMES" \
    --resize-scale "$RESIZE_SCALE"
fi

if [[ $SKIP_COLMAP -eq 0 ]]; then
  echo "==> Running COLMAP calibration"
  bash scripts/run_colmap_calib.sh \
    --data-dir "$DATA_DIR" \
    --num-cameras "$NUM_CAMERAS" \
    --image-ext png
fi

if [[ $SKIP_ROMA -eq 0 ]]; then
  echo "==> ROMA triangulation to NPY"
  if [[ "${KMP_DUPLICATE_LIB_OK:-}" != "TRUE" ]]; then
    echo "[NOTICE] Setting KMP_DUPLICATE_LIB_OK=TRUE to avoid libomp double-init crash on macOS."
    export KMP_DUPLICATE_LIB_OK=TRUE
  fi
  if [[ "$ROMA_PER_FRAME" -eq 1 ]]; then
    for ((frame=0; frame<NUM_FRAMES; frame++)); do
      python3 scripts/roma_triangulate_to_npy.py "${ROMA_ARGS[@]}" --frame-start "$frame" --frame-end "$frame"
    done
  else
    python3 scripts/roma_triangulate_to_npy.py "${ROMA_ARGS[@]}" --frame-start 0 --frame-end $((NUM_FRAMES - 1))
  fi
fi

if [[ $SKIP_COMBINE -eq 0 ]]; then
  echo "==> Combining keyframes"
  python3 src/combine_frames_fast_keyframes.py \
    --input-dir "$TRIANGULATION_DIR" \
    --output-path "$NPZ_PATH" \
    --frame-start 0 \
    --frame-end $((NUM_FRAMES - 1)) \
    --keyframe-step "$KEYFRAME_STEP"
fi

echo "======================================"
echo "Data dir: $DATA_DIR"
echo "Images: $IMAGES_DIR"
echo "COLMAP: $COLMAP_DIR/sparse/0"
echo "Triangulation: $TRIANGULATION_DIR"
echo "NPZ: $NPZ_PATH"
echo "======================================"