#!/bin/bash
set -euo pipefail

VIDEO_DIR="${VIDEO_DIR:-$(pwd)/dance}"
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/dataset/selfcap_dance/images}"
NUM_CAMERAS="${NUM_CAMERAS:-24}"
NUM_FRAMES="${NUM_FRAMES:-60}"
RESIZE_SCALE="${RESIZE_SCALE:-0.5}"

usage() {
  echo "Usage: $0 [--video-dir PATH] [--output-dir PATH] [--num-cameras N] [--num-frames N] [--resize-scale S]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir)
      VIDEO_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num-cameras)
      NUM_CAMERAS="$2"
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="$2"
      shift 2
      ;;
    --resize-scale)
      RESIZE_SCALE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "[ERROR] video dir not found: $VIDEO_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

SCALE_FILTER=""
if [[ "$RESIZE_SCALE" != "1" && "$RESIZE_SCALE" != "1.0" ]]; then
  SCALE_FILTER="-vf scale=iw*${RESIZE_SCALE}:ih*${RESIZE_SCALE}"
fi

echo "==> Extracting frames"
echo "  Video dir: $VIDEO_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Cameras: $NUM_CAMERAS"
echo "  Frames: $NUM_FRAMES"
echo "  Resize scale: $RESIZE_SCALE"

for ((i=0; i<NUM_CAMERAS; i++)); do
  cam=$(printf "%04d" "$i")
  video_path="$VIDEO_DIR/${cam}.mp4"
  if [[ ! -f "$video_path" ]]; then
    echo "[ERROR] missing video: $video_path"
    exit 1
  fi

  cam_dir="$OUTPUT_DIR/$cam"
  mkdir -p "$cam_dir"

  first_frame="$cam_dir/000000.png"
  last_frame=$(printf "%s/%06d.png" "$cam_dir" $((NUM_FRAMES - 1)))
  if [[ -f "$first_frame" && -f "$last_frame" ]]; then
    echo "[NOTICE] Frames already exist for camera $cam. Skipping ffmpeg extraction."
    continue
  fi

  ffmpeg -y \
    -i "$video_path" \
    $SCALE_FILTER \
    -frames:v "$NUM_FRAMES" \
    -vsync 0 \
    -start_number 0 \
    "$cam_dir/%06d.png"

done

echo "==> Done: extracted frames to $OUTPUT_DIR"