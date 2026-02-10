#!/bin/bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-$(pwd)/dataset/dance}"
# CALIB_DIR, COLMAP_DIR は引数パース後に DATA_DIR から導出する
_CLI_CALIB_DIR=""
_CLI_COLMAP_DIR=""
NUM_CAMERAS="${NUM_CAMERAS:-24}"
IMAGE_EXT="${IMAGE_EXT:-png}"

usage() {
  echo "Usage: $0 [--data-dir PATH] [--calib-dir PATH] [--colmap-dir PATH] [--num-cameras N] [--image-ext EXT]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --calib-dir)
      _CLI_CALIB_DIR="$2"
      shift 2
      ;;
    --colmap-dir)
      _CLI_COLMAP_DIR="$2"
      shift 2
      ;;
    --num-cameras)
      NUM_CAMERAS="$2"
      shift 2
      ;;
    --image-ext)
      IMAGE_EXT="$2"
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

# --data-dir の更新を反映して派生パスを確定する
CALIB_DIR="${_CLI_CALIB_DIR:-${CALIB_DIR:-$DATA_DIR/calib_images}}"
COLMAP_DIR="${_CLI_COLMAP_DIR:-${COLMAP_DIR:-$DATA_DIR/colmap}}"

IMAGES_DIR="$DATA_DIR/images"
DB_PATH="$COLMAP_DIR/database.db"
SPARSE_DIR="$COLMAP_DIR/sparse"
SPARSE_MODEL="$SPARSE_DIR/0"

mkdir -p "$CALIB_DIR" "$COLMAP_DIR" "$SPARSE_DIR"

if [[ -f "$SPARSE_MODEL/cameras.bin" || -f "$SPARSE_MODEL/cameras.txt" ]]; then
  echo "[NOTICE] COLMAP calibration already exists at $SPARSE_MODEL. Skipping."
  exit 0
fi

if [[ ! -d "$IMAGES_DIR" ]]; then
  echo "[ERROR] images dir not found: $IMAGES_DIR"
  exit 1
fi

# Create calibration images from the first frame of each camera
for ((i=0; i<NUM_CAMERAS; i++)); do
  cam=$(printf "%04d" "$i")
  src_frame="$IMAGES_DIR/$cam/000000.${IMAGE_EXT}"
  dst_frame="$CALIB_DIR/${cam}.png"
  if [[ ! -f "$src_frame" ]]; then
    echo "[ERROR] missing first frame: $src_frame"
    exit 1
  fi
  cp "$src_frame" "$dst_frame"
done

echo "==> COLMAP feature extraction"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$CALIB_DIR" \
  --ImageReader.single_camera 0 \
  --ImageReader.camera_model OPENCV \
  --FeatureExtraction.use_gpu 1 \
  --SiftExtraction.max_num_features 12000 \
  --SiftExtraction.peak_threshold 0.01

echo "==> COLMAP exhaustive matching"
colmap exhaustive_matcher \
  --database_path "$DB_PATH" \
  --FeatureMatching.use_gpu 1

echo "==> COLMAP mapping"
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$CALIB_DIR" \
  --output_path "$SPARSE_DIR" \
  --Mapper.init_min_num_inliers 200 \
  --Mapper.ba_global_max_num_iterations 50

echo "==> Done: COLMAP model at $SPARSE_DIR/0"