#!/bin/bash
# SelfCap Dance データ準備パイプライン
# 4つのステップを順に実行する:
#   1. フレーム抽出（MP4 → PNG）
#   2. COLMAPキャリブレーション（カメラ内部/外部パラメータ推定）
#   3. RoMa三角測量（マルチビューマッチング → 3D点群NPY）
#   4. キーフレーム結合（NPY群 → NPZ）
set -euo pipefail

# ===========================
# デフォルトパラメータ
# ===========================

# --- 基本パラメータ ---
VIDEO_DIR="${VIDEO_DIR:-$(pwd)/dance}"                # 入力動画ディレクトリ（CCCC.mp4）
DATA_DIR="${DATA_DIR:-$(pwd)/dataset/selfcap_dance}"  # データ出力ルート
NUM_CAMERAS="${NUM_CAMERAS:-24}"    # カメラ台数
NUM_FRAMES="${NUM_FRAMES:-60}"     # フレーム数（60fps前提で1秒分）
FPS="${FPS:-60}"                   # 抽出FPS（4K 60fps動画からの抽出を想定）
RESIZE_SCALE="${RESIZE_SCALE:-1}"  # リサイズ倍率（1=4K原寸維持、0.5で半分）
KEYFRAME_STEP="${KEYFRAME_STEP:-5}"  # キーフレーム間隔（速度推定用）

# --- RoMa三角測量パラメータ ---
ROMA_REF_CAM="${ROMA_REF_CAM:-0000}"       # リファレンスカメラID
ROMA_DEVICE="${ROMA_DEVICE:-auto}"         # GPUデバイス（auto/cuda/mps）
ROMA_CERTAINTY="${ROMA_CERTAINTY:-0.3}"    # マッチング確信度しきい値
ROMA_MIN_DEPTH="${ROMA_MIN_DEPTH:-1e-4}"   # 最小深度フィルタ
ROMA_IMAGE_SCALE="${ROMA_IMAGE_SCALE:-1.0}"  # RoMa入力画像のスケーリング
ROMA_AMP="${ROMA_AMP:-0}"                  # AMP (FP16混合精度) 0=無効
ROMA_USE_RANSAC="${ROMA_USE_RANSAC:-0}"    # RANSAC外れ値除去 0=無効
ROMA_RANSAC_TH="${ROMA_RANSAC_TH:-0.5}"   # RANSACピクセル誤差しきい値
ROMA_VOXEL_SIZE="${ROMA_VOXEL_SIZE:-0}"    # ボクセルダウンサンプリング（0=無効）
ROMA_NO_UPSAMPLE="${ROMA_NO_UPSAMPLE:-0}" # RoMaアップサンプリング無効化（0=有効/精度優先）
# 一括処理をデフォルトにする: gc.collect() + empty_cache() でVRAMリーク対策済みのため
# プロセスごとの分離は不要（RTX 5090 32GBで60フレーム一括処理可能）
ROMA_PER_FRAME="${ROMA_PER_FRAME:-0}"      # 0=一括処理（推奨）, 1=フレーム毎にプロセス分離（非推奨・後方互換用）

# --- ステップスキップフラグ ---
SKIP_EXTRACT="${SKIP_EXTRACT:-0}"   # 1でフレーム抽出をスキップ
SKIP_COLMAP="${SKIP_COLMAP:-0}"     # 1でCOLMAPキャリブレーションをスキップ
SKIP_ROMA="${SKIP_ROMA:-0}"         # 1でRoMa三角測量をスキップ
SKIP_COMBINE="${SKIP_COMBINE:-0}"   # 1でキーフレーム結合をスキップ

# --- ヘルプ表示 ---
usage() {
  echo "Usage: $0 [--video-dir PATH] [--data-dir PATH] [--num-cameras N] [--num-frames N]"
  echo "          [--fps N] [--resize-scale S] [--keyframe-step N]"
  echo "          [--roma-ref-cam ID] [--roma-device DEV] [--roma-certainty TH]"
  echo "          [--roma-min-depth D] [--roma-image-scale S] [--roma-amp]"
  echo "          [--roma-use-ransac] [--roma-ransac-th PX] [--roma-voxel-size M]"
  echo "          [--roma-no-upsample] [--roma-per-frame 0|1]"
  echo "          [--skip-extract] [--skip-colmap] [--skip-roma] [--skip-combine]"
}

# ===========================
# コマンドライン引数のパース
# ===========================
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir)
      VIDEO_DIR="$2"; shift 2 ;;       # 入力動画ディレクトリ
    --data-dir)
      DATA_DIR="$2"; shift 2 ;;        # データ出力ルート
    --num-cameras)
      NUM_CAMERAS="$2"; shift 2 ;;     # カメラ台数
    --num-frames)
      NUM_FRAMES="$2"; shift 2 ;;      # フレーム数
    --fps)
      FPS="$2"; shift 2 ;;             # 抽出FPS
    --resize-scale)
      RESIZE_SCALE="$2"; shift 2 ;;    # リサイズ倍率
    --keyframe-step)
      KEYFRAME_STEP="$2"; shift 2 ;;   # キーフレーム間隔
    --roma-ref-cam)
      ROMA_REF_CAM="$2"; shift 2 ;;    # リファレンスカメラID
    --roma-device)
      ROMA_DEVICE="$2"; shift 2 ;;     # GPUデバイス
    --roma-certainty)
      ROMA_CERTAINTY="$2"; shift 2 ;;  # マッチング確信度しきい値
    --roma-min-depth)
      ROMA_MIN_DEPTH="$2"; shift 2 ;;  # 最小深度フィルタ
    --roma-image-scale)
      ROMA_IMAGE_SCALE="$2"; shift 2 ;; # RoMa入力画像スケーリング
    --roma-amp)
      ROMA_AMP=1; shift ;;             # AMP有効化
    --roma-use-ransac)
      ROMA_USE_RANSAC=1; shift ;;      # RANSAC有効化
    --roma-ransac-th)
      ROMA_RANSAC_TH="$2"; shift 2 ;;  # RANSACしきい値
    --roma-voxel-size)
      ROMA_VOXEL_SIZE="$2"; shift 2 ;;  # ボクセルサイズ
    --roma-no-upsample)
      ROMA_NO_UPSAMPLE=1; shift ;;     # RoMaアップサンプリング無効化
    --roma-per-frame)
      ROMA_PER_FRAME="$2"; shift 2 ;;  # フレーム毎プロセス分離（後方互換用）
    --skip-extract)
      SKIP_EXTRACT=1; shift ;;         # フレーム抽出スキップ
    --skip-colmap)
      SKIP_COLMAP=1; shift ;;          # COLMAPスキップ
    --skip-roma)
      SKIP_ROMA=1; shift ;;            # RoMa三角測量スキップ
    --skip-combine)
      SKIP_COMBINE=1; shift ;;         # キーフレーム結合スキップ
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

# ===========================
# 派生パスの構築
# ===========================
IMAGES_DIR="$DATA_DIR/images"                    # カメラ別画像ディレクトリ
COLMAP_DIR="$DATA_DIR/colmap"                    # COLMAPモデルディレクトリ
TRIANGULATION_DIR="$DATA_DIR/triangulation"      # 三角測量結果の出力先
NPZ_PATH="$DATA_DIR/keyframes_${NUM_FRAMES}frames_step${KEYFRAME_STEP}.npz"  # 最終NPZファイル

# --- RoMa三角測量の引数配列を構築 ---
ROMA_ARGS=(
  --images-dir "$IMAGES_DIR"                # カメラ別画像ディレクトリ
  --colmap-model "$COLMAP_DIR/sparse/0"     # COLMAPスパースモデル
  --output-dir "$TRIANGULATION_DIR"         # NPY出力先
  --frame-step 1                            # 全フレームを処理（三角測量はstep=1が基本）
  --ref-cam "$ROMA_REF_CAM"                # リファレンスカメラID
  --device "$ROMA_DEVICE"                   # GPUデバイス
  --certainty "$ROMA_CERTAINTY"             # 確信度しきい値
  --min-depth "$ROMA_MIN_DEPTH"             # 最小深度フィルタ
  --image-scale "$ROMA_IMAGE_SCALE"         # 入力画像スケーリング
  --voxel-size "$ROMA_VOXEL_SIZE"           # ボクセルダウンサンプリング
)

# AMP有効の場合、引数に追加
if [[ $ROMA_AMP -eq 1 ]]; then
  ROMA_ARGS+=(--amp)
fi
# RANSAC有効の場合、引数に追加
if [[ $ROMA_USE_RANSAC -eq 1 ]]; then
  ROMA_ARGS+=(--use-ransac --ransac-th "$ROMA_RANSAC_TH")
fi
# RoMaアップサンプリング無効化の場合、引数に追加
# (解像度 864→560 に下がるがVRAMを大幅に削減)
if [[ $ROMA_NO_UPSAMPLE -eq 1 ]]; then
  ROMA_ARGS+=(--no-upsample)
fi

# データ出力ルートディレクトリを作成
mkdir -p "$DATA_DIR"

# ===========================
# Step 1: フレーム抽出（MP4 → PNG）
# ===========================
if [[ $SKIP_EXTRACT -eq 0 ]]; then
  echo "==> Extracting frames (run inside tmux recommended)"
  bash scripts/extract_selfcap_frames.sh \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$IMAGES_DIR" \
    --num-cameras "$NUM_CAMERAS" \
    --num-frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --resize-scale "$RESIZE_SCALE"
fi

# ===========================
# Step 2: COLMAPキャリブレーション
# ===========================
if [[ $SKIP_COLMAP -eq 0 ]]; then
  echo "==> Running COLMAP calibration"
  bash scripts/run_colmap_calib.sh \
    --data-dir "$DATA_DIR" \
    --num-cameras "$NUM_CAMERAS" \
    --image-ext png
fi

# ===========================
# Step 3: RoMa三角測量 → per-frame NPY生成
# ===========================
if [[ $SKIP_ROMA -eq 0 ]]; then
  echo "==> ROMA triangulation to NPY"

  # MPS/auto デバイスの場合、CPUフォールバックを有効化
  # （一部のPyTorch opsがMPS未対応のため）
  if [[ "$ROMA_DEVICE" == "mps" || "$ROMA_DEVICE" == "auto" ]]; then
    if [[ "${PYTORCH_ENABLE_MPS_FALLBACK:-}" != "1" ]]; then
      echo "[NOTICE] Setting PYTORCH_ENABLE_MPS_FALLBACK=1 to enable CPU fallback for unsupported MPS ops."
      export PYTORCH_ENABLE_MPS_FALLBACK=1
    fi
  fi

  # macOSでlibompの二重初期化クラッシュを回避
  if [[ "${KMP_DUPLICATE_LIB_OK:-}" != "TRUE" ]]; then
    echo "[NOTICE] Setting KMP_DUPLICATE_LIB_OK=TRUE to avoid libomp double-init crash on macOS."
    export KMP_DUPLICATE_LIB_OK=TRUE
  fi

  if [[ "$ROMA_PER_FRAME" -eq 1 ]]; then
    # 非推奨: フレーム毎にプロセスを分離して実行（後方互換用）
    # 毎回RoMaモデルを再ロードするため ~10秒/フレーム のオーバーヘッドが発生する
    echo "[WARN] ROMA_PER_FRAME=1 is deprecated. Use default (0) for better performance."
    for ((frame=0; frame<NUM_FRAMES; frame++)); do
      python3 scripts/roma_triangulate_to_npy.py "${ROMA_ARGS[@]}" --frame-start "$frame" --frame-end "$frame"
    done
  else
    # 推奨: 全フレームを一括処理（RoMaモデルを1回だけロード）
    # gc.collect() + empty_cache() によるVRAMリーク対策済みのため、
    # RTX 5090 (VRAM 32GB) で60フレーム一括処理が可能
    python3 scripts/roma_triangulate_to_npy.py "${ROMA_ARGS[@]}" --frame-start 0 --frame-end $((NUM_FRAMES - 1))
  fi
fi

# ===========================
# Step 4: キーフレーム結合 → NPZ
# ===========================
if [[ $SKIP_COMBINE -eq 0 ]]; then
  echo "==> Combining keyframes"
  python3 src/combine_frames_fast_keyframes.py \
    --input-dir "$TRIANGULATION_DIR" \
    --output-path "$NPZ_PATH" \
    --frame-start 0 \
    --frame-end $((NUM_FRAMES - 1)) \
    --keyframe-step "$KEYFRAME_STEP"
fi

# ===========================
# パイプライン完了サマリー
# ===========================
echo "======================================"
echo "Data dir: $DATA_DIR"
echo "Images: $IMAGES_DIR"
echo "COLMAP: $COLMAP_DIR/sparse/0"
echo "Triangulation: $TRIANGULATION_DIR"
echo "NPZ: $NPZ_PATH"
echo "======================================"
