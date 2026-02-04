#!/bin/bash
set -e

# Usage:
#   bash scripts/run_roma_triangulation.sh <images_dir> <colmap_model_dir> <output_dir> <start_frame> <end_frame> [ref_cam]

IMAGES_DIR=$1
COLMAP_MODEL=$2
OUTPUT_DIR=$3
START_FRAME=$4
END_FRAME=$5
REF_CAM=${6:-0000}

if [ -z "$IMAGES_DIR" ] || [ -z "$COLMAP_MODEL" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$START_FRAME" ] || [ -z "$END_FRAME" ]; then
  echo "Usage: bash scripts/run_roma_triangulation.sh <images_dir> <colmap_model_dir> <output_dir> <start_frame> <end_frame> [ref_cam]"
  exit 1
fi

source .venv/bin/activate

python3 scripts/roma_triangulate_to_npy.py \
  --images-dir "$IMAGES_DIR" \
  --colmap-model "$COLMAP_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --frame-start "$START_FRAME" \
  --frame-end "$END_FRAME" \
  --ref-cam "$REF_CAM"
