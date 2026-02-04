#!/bin/bash
set -e

#######################################
# 設定（デフォルト）
#######################################
ROOT_DIR="$(pwd)/dance"
FRAME_TIME="0"   # 先頭フレーム（秒）
NUM_CAMERAS=24
NUM_FRAMES=60
SKIP_COLMAP=0
RUN_ROMA=1
RUN_COMBINE=1
KEYFRAME_STEP=1
ROMA_REF_CAM="0000"
ROMA_DEVICE="cuda"
ROMA_CERTAINTY=0.3
ROMA_USE_RANSAC=0
ROMA_RANSAC_TH=0.5
ROMA_MIN_DEPTH=1e-4
TRIANGULATION_DIR=""
NPZ_PATH=""

usage() {
  echo "Usage: $0 [--root-dir PATH] [--frame-time SEC] [--num-cameras N] [--num-frames N] [--skip-colmap]"
  echo "          [--run-roma] [--run-combine] [--keyframe-step N]"
  echo "          [--triangulation-dir PATH] [--npz-path PATH]"
  echo "          [--roma-ref-cam ID] [--roma-device DEV] [--roma-certainty TH]"
  echo "          [--roma-use-ransac] [--roma-ransac-th PX] [--roma-min-depth D]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root-dir)
      ROOT_DIR="$2"
      shift 2
      ;;
    --frame-time)
      FRAME_TIME="$2"
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
    --skip-colmap)
      SKIP_COLMAP=1
      shift
      ;;
    --run-roma)
      RUN_ROMA=1
      shift
      ;;
    --run-combine)
      RUN_COMBINE=1
      shift
      ;;
    --keyframe-step)
      KEYFRAME_STEP="$2"
      shift 2
      ;;
    --triangulation-dir)
      TRIANGULATION_DIR="$2"
      shift 2
      ;;
    --npz-path)
      NPZ_PATH="$2"
      shift 2
      ;;
    --roma-ref-cam)
      ROMA_REF_CAM="$2"
      shift 2
      ;;
    --roma-device)
      ROMA_DEVICE="$2"
      shift 2
      ;;
    --roma-certainty)
      ROMA_CERTAINTY="$2"
      shift 2
      ;;
    --roma-use-ransac)
      ROMA_USE_RANSAC=1
      shift
      ;;
    --roma-ransac-th)
      ROMA_RANSAC_TH="$2"
      shift 2
      ;;
    --roma-min-depth)
      ROMA_MIN_DEPTH="$2"
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

VIDEO_DIR="$ROOT_DIR"
IMG_DIR="$ROOT_DIR/calib_images"
IMAGES_DIR="$ROOT_DIR/images"
COLMAP_DIR="$ROOT_DIR/colmap"
DB_PATH="$COLMAP_DIR/database.db"
SPARSE_DIR="$COLMAP_DIR/sparse"
SRC_SPARSE="$SPARSE_DIR/0"
DST_SPARSE="$COLMAP_DIR/sparse_all"
DEFAULT_TRIANGULATION_DIR="$ROOT_DIR/triangulation"
DEFAULT_NPZ_PATH="$ROOT_DIR/keyframes_${NUM_FRAMES}frames_step${KEYFRAME_STEP}.npz"

if [[ -z "$TRIANGULATION_DIR" ]]; then
  TRIANGULATION_DIR="$DEFAULT_TRIANGULATION_DIR"
fi

if [[ -z "$NPZ_PATH" ]]; then
  NPZ_PATH="$DEFAULT_NPZ_PATH"
fi

ROMA_ARGS=(
  --images-dir "$IMAGES_DIR"
  --colmap-model "$SRC_SPARSE"
  --output-dir "$TRIANGULATION_DIR"
  --frame-start 0
  --frame-end $((NUM_FRAMES - 1))
  --frame-step 1
  --ref-cam "$ROMA_REF_CAM"
  --device "$ROMA_DEVICE"
  --certainty "$ROMA_CERTAINTY"
  --min-depth "$ROMA_MIN_DEPTH"
)

if [[ $ROMA_USE_RANSAC -eq 1 ]]; then
  ROMA_ARGS+=(--use-ransac --ransac-th "$ROMA_RANSAC_TH")
fi

#######################################
# ディレクトリ作成
#######################################
mkdir -p "$IMG_DIR"
mkdir -p "$SPARSE_DIR"

#######################################
# 1. 動画 → 画像（1フレームのみ）
#######################################
if [[ $SKIP_COLMAP -eq 0 ]]; then
  echo "==> Extracting calibration frames..."

  for i in $(seq -f "%04g" 0 $((NUM_CAMERAS-1))); do
    ffmpeg -y \
      -ss "$FRAME_TIME" \
      -i "$VIDEO_DIR/$i.mp4" \
      -frames:v 1 \
      "$IMG_DIR/cam$i.png"
  done

#######################################
# 2. COLMAP: Feature Extraction
#######################################
  echo "==> COLMAP feature extraction..."

  colmap feature_extractor \
    --database_path "$DB_PATH" \
    --image_path "$IMG_DIR" \
    --ImageReader.single_camera 0 \
    --FeatureExtraction.use_gpu 1 \
    --SiftExtraction.max_num_features 12000 \
    --SiftExtraction.peak_threshold 0.01

#######################################
# 3. COLMAP: Matching（24台なので exhaustive OK）
#######################################
  echo "==> COLMAP matching..."

  colmap exhaustive_matcher \
    --database_path "$DB_PATH" \
    --FeatureMatching.use_gpu 1

#######################################
# 4. COLMAP: Mapper（SFM）
#######################################
  echo "==> COLMAP mapping..."

  colmap mapper \
    --database_path "$DB_PATH" \
    --image_path "$IMG_DIR" \
    --output_path "$SPARSE_DIR" \
    --Mapper.init_min_num_inliers 200 \
    --Mapper.ba_global_max_num_iterations 50
fi

#######################################
# 5. Sparse 展開（frames/points3D 作成）
#######################################
mkdir -p "$DST_SPARSE"

echo "==> Convert cameras.bin to cameras.txt"
colmap model_converter \
  --input_path "$SRC_SPARSE" \
  --output_path "$DST_SPARSE" \
  --output_type TXT

echo "==> Expanding images to all frames"
python3 datasets/expand_images.py \
  --src "$SRC_SPARSE/images.bin" \
  --dst "$DST_SPARSE/images.txt" \
  --num_frames $NUM_FRAMES \
  --num_cameras $NUM_CAMERAS

echo "==> Creating empty points3D.txt"
: > "$DST_SPARSE/points3D.txt"

#######################################
# 6. RoMa triangulation（任意）
#######################################
if [[ $RUN_ROMA -eq 1 ]]; then
  if [[ ! -d "$IMAGES_DIR" ]]; then
    mkdir -p "$IMAGES_DIR"
  fi

  shopt -s nullglob
  images_found=("$IMAGES_DIR"/*.png "$IMAGES_DIR"/*.jpg "$IMAGES_DIR"/*.jpeg)
  shopt -u nullglob

  if [[ ${#images_found[@]} -eq 0 ]]; then
    echo "==> No per-frame images found. Extracting from videos..."

    for i in $(seq -f "%04g" 0 $((NUM_CAMERAS-1))); do
      video_path="$VIDEO_DIR/$i.mp4"
      if [[ ! -f "$video_path" ]]; then
        echo "[ERROR] missing video: $video_path"
        exit 1
      fi

      ffmpeg -y \
        -i "$video_path" \
        -frames:v "$NUM_FRAMES" \
        -vsync 0 \
        -start_number 0 \
        "$IMAGES_DIR/${i}_frame%06d.png"
    done
  fi

  if [[ ! -d "$ROOT_DIR/sparse/0" ]]; then
    mkdir -p "$ROOT_DIR/sparse"
    ln -s "$SRC_SPARSE" "$ROOT_DIR/sparse/0"
  fi

  echo "==> RoMa triangulation to NPY"


  python3 scripts/roma_triangulate_to_npy.py "${ROMA_ARGS[@]}"
fi

#######################################
# 7. Combine keyframes（任意）
#######################################
if [[ $RUN_COMBINE -eq 1 ]]; then
  echo "==> Combining keyframes with velocity"
  python3 src/combine_frames_fast_keyframes.py \
    --input-dir "$TRIANGULATION_DIR" \
    --output-path "$NPZ_PATH" \
    --frame-start 0 \
    --frame-end $((NUM_FRAMES - 1)) \
    --keyframe-step "$KEYFRAME_STEP"
fi

#######################################
# 完了
#######################################
echo "======================================"
if [[ $SKIP_COLMAP -eq 0 ]]; then
  echo "COLMAP calibration finished successfully"
  echo "Sparse model:"
  echo "  $SPARSE_DIR/0"
fi
echo "Sparse model expanded successfully"
echo "Output: $DST_SPARSE"
if [[ $RUN_ROMA -eq 1 ]]; then
  echo "Triangulation output: $TRIANGULATION_DIR"
fi
if [[ $RUN_COMBINE -eq 1 ]]; then
  echo "NPZ output: $NPZ_PATH"
fi
echo "======================================"
