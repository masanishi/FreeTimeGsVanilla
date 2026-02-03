#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------
# 多視点動画から FreeTimeGS 用の COLMAP データを作るスクリプト
#
# 使い方:
#   ./scripts/prepare_dataset.sh <videos_dir> <data_dir> <frame_start> <frame_end> [fps] [downsample] [skip_extract]
#
# 例:
#   ./scripts/prepare_dataset.sh ./dance ./dataset 0 60 60 2
#   ./scripts/prepare_dataset.sh ./dance ./dataset 0 60 60 2 true
#   ※ downsample=2 の場合、幅/高さを 1/2 に縮小します
#   ※ skip_extract=true の場合、フレーム抽出をスキップします
#
# 前提:
# - videos_dir には cam01.mp4, cam02.mp4, ... のようにカメラ名付き動画がある
# - すべての動画は同じFPS・同時刻開始で撮影されている
#
# 出力:
#   data_dir/images/  に各カメラのフレーム画像（png）
#   data_dir/sparse/0 に COLMAP の再構成結果
# ---------------------------------------------

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "Usage: $0 <videos_dir> <data_dir> <frame_start> <frame_end> [fps] [downsample] [skip_extract]"
  exit 1
fi

VIDEOS_DIR="$1"
DATA_DIR="$2"
FRAME_START="$3"
FRAME_END="$4"
FPS="${5:-30}"
DOWNSAMPLE="${6:-1}"
SKIP_EXTRACT="${7:-false}"
GPU_INDEX="${GPU_INDEX:-0}"

IMAGES_DIR="$DATA_DIR/images"
SPARSE_DIR="$DATA_DIR/sparse"
DB_PATH="$DATA_DIR/database.db"

# ディレクトリを作成
mkdir -p "$IMAGES_DIR"
mkdir -p "$SPARSE_DIR"

# ---------------------------------------------
# 1) ffmpeg で各カメラ動画をフレーム画像に分解
#    - 同じフレーム番号が同時刻になるように抽出
#    - 抽出範囲は frame_start 〜 frame_end
# ---------------------------------------------

if [[ "${SKIP_EXTRACT}" == "true" ]]; then
  echo "[INFO] フレーム抽出をスキップします (skip_extract=true)"
else
  shopt -s nullglob
  VIDEO_FILES=("$VIDEOS_DIR"/*.mp4)
  if [[ ${#VIDEO_FILES[@]} -eq 0 ]]; then
    echo "[ERROR] $VIDEOS_DIR に mp4 が見つかりません"
    exit 1
  fi

  for video_path in "${VIDEO_FILES[@]}"; do
    video_file="$(basename "$video_path")"
    cam_name="${video_file%.*}"

    echo "[INFO] 抽出開始: $video_file"

    # フレーム番号は抽出範囲内で 000000 から振り直します。
    # 全カメラで同じ範囲を切るので、同時刻は一致します。
    VF_FILTER="fps=${FPS},select='between(n,${FRAME_START},${FRAME_END})'"
    if [[ "$DOWNSAMPLE" != "1" ]]; then
      VF_FILTER+=" ,scale=iw/${DOWNSAMPLE}:ih/${DOWNSAMPLE}"
    fi

    ffmpeg -y \
      -i "$video_path" \
      -vf "$VF_FILTER" \
      -vsync 0 \
      -start_number 0 \
      "$IMAGES_DIR/${cam_name}_frame%06d.png"

  done
fi

# ---------------------------------------------
# 2) COLMAP で Sparse Reconstruction を作成
# ---------------------------------------------

# 既存DBがある場合は削除（再実行のため）
if [[ -f "$DB_PATH" ]]; then
  echo "[INFO] 既存の database.db を削除"
  rm -f "$DB_PATH"
fi

# データベース作成
colmap database_creator \
  --database_path "$DB_PATH"

# 特徴抽出
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --ImageReader.single_camera 1 \
  --SiftExtraction.use_gpu 1 \
  --SiftExtraction.gpu_index "$GPU_INDEX"

# 画像マッチング（全画像を総当たり）
colmap exhaustive_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu 1 \
  --SiftMatching.gpu_index "$GPU_INDEX"

# 三角測量（Sparse Reconstruction）
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$IMAGES_DIR" \
  --output_path "$SPARSE_DIR"

# ---------------------------------------------
# 3) 結果確認
# ---------------------------------------------

if [[ -f "$SPARSE_DIR/0/cameras.bin" && -f "$SPARSE_DIR/0/images.bin" && -f "$SPARSE_DIR/0/points3D.bin" ]]; then
  echo "[SUCCESS] COLMAP データ準備完了: $SPARSE_DIR/0"
  echo "          FreeTime_dataset.py を実行できる状態です。"
else
  echo "[ERROR] COLMAP の出力が見つかりません。ログを確認してください。"
  exit 1
fi
