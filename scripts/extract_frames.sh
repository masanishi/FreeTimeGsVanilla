#!/bin/bash
# MP4動画からPNGフレームを抽出するスクリプト
# 各カメラ（0000.mp4, 0001.mp4, ...）からフレームを切り出し、
# images/CCCC/FFFFFF.png の形式で保存する
set -euo pipefail

# --- デフォルトパラメータ ---
VIDEO_DIR="${VIDEO_DIR:-$(pwd)/movies/dance}"     # 入力動画ディレクトリ
OUTPUT_DIR="${OUTPUT_DIR:-$(pwd)/dataset/dance/images}"  # 出力画像ディレクトリ
NUM_CAMERAS="${NUM_CAMERAS:-24}"           # カメラ台数
NUM_FRAMES="${NUM_FRAMES:-60}"            # 抽出フレーム数
START_FRAME="${START_FRAME:-0}"           # 抽出開始フレーム番号（0=先頭から）
RESIZE_SCALE="${RESIZE_SCALE:-0.5}"       # リサイズ倍率（0.5=論文準拠、1920x1080。1=4K原寸維持）
FPS="${FPS:-60}"                          # 抽出FPS（元動画のFPSに依存せず指定FPSで抽出）

# --- ヘルプ表示 ---
usage() {
  echo "Usage: $0 [--video-dir PATH] [--output-dir PATH] [--num-cameras N] [--num-frames N] [--start-frame N] [--resize-scale S] [--fps N]"
}

# --- コマンドライン引数のパース ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --video-dir)
      VIDEO_DIR="$2"    # 動画ファイルが格納されたディレクトリ
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"   # 抽出フレームの出力先
      shift 2
      ;;
    --num-cameras)
      NUM_CAMERAS="$2"  # 処理するカメラ台数（0000～NUM_CAMERAS-1）
      shift 2
      ;;
    --num-frames)
      NUM_FRAMES="$2"   # 各カメラから抽出するフレーム数
      shift 2
      ;;
    --start-frame)
      START_FRAME="$2"  # 抽出開始フレーム番号（0=先頭）
      shift 2
      ;;
    --resize-scale)
      RESIZE_SCALE="$2" # リサイズ倍率（0.5で半分、1で原寸）
      shift 2
      ;;
    --fps)
      FPS="$2"          # 抽出FPS（-vf fps=N で指定）
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

# --- 入力ディレクトリの存在チェック ---
if [[ ! -d "$VIDEO_DIR" ]]; then
  echo "[ERROR] video dir not found: $VIDEO_DIR"
  exit 1
fi

# 出力ディレクトリを作成
mkdir -p "$OUTPUT_DIR"

# --- ffmpegフィルタの構築 ---
# fps=N フィルタ: 元動画のFPSに関係なく、指定FPSでフレームを抽出する
# scale フィルタ: RESIZE_SCALE != 1 の場合のみ追加（4K原寸ならscaleは不要）
VF_FILTERS="fps=${FPS}"  # ベースフィルタ: 常にfpsフィルタを適用
if [[ "$RESIZE_SCALE" != "1" && "$RESIZE_SCALE" != "1.0" ]]; then
  # fpsフィルタの後にscaleフィルタをカンマ連結で追加
  VF_FILTERS="${VF_FILTERS},scale=iw*${RESIZE_SCALE}:ih*${RESIZE_SCALE}"
fi

# --- パラメータのログ表示 ---
echo "==> Extracting frames"
echo "  Video dir: $VIDEO_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  Cameras: $NUM_CAMERAS"
echo "  Start frame: $START_FRAME"
echo "  Frames: $NUM_FRAMES"
echo "  FPS: $FPS"
echo "  Resize scale: $RESIZE_SCALE"

# --- 開始フレームからシーク時間を計算 ---
# START_FRAME > 0 の場合、ffmpegの -ss でその位置までシークする
# 計算: start_frame / fps = 秒数
if [[ "$START_FRAME" -gt 0 ]]; then
  # bcで浮動小数点計算（zsh/bash両対応）
  SEEK_SECONDS=$(echo "scale=6; $START_FRAME / $FPS" | bc)
  SEEK_OPT="-ss $SEEK_SECONDS"
  echo "  Seek: ${SEEK_SECONDS}s (frame $START_FRAME at ${FPS}fps)"
else
  SEEK_OPT=""
fi

# --- カメラごとのフレーム抽出ループ ---
for ((i=0; i<NUM_CAMERAS; i++)); do
  # カメラ名を4桁ゼロ埋め（0000, 0001, ..., 0023）
  cam=$(printf "%04d" "$i")
  video_path="$VIDEO_DIR/${cam}.mp4"  # 入力動画パス

  # 動画ファイルの存在チェック
  if [[ ! -f "$video_path" ]]; then
    echo "[ERROR] missing video: $video_path"
    exit 1
  fi

  # カメラ別の出力ディレクトリを作成
  cam_dir="$OUTPUT_DIR/$cam"
  mkdir -p "$cam_dir"

  # --- 抽出済みチェック: 最初と最後のフレームが両方存在すればスキップ ---
  first_frame="$cam_dir/000000.png"
  last_frame=$(printf "%s/%06d.png" "$cam_dir" $((NUM_FRAMES - 1)))
  if [[ -f "$first_frame" && -f "$last_frame" ]]; then
    echo "[NOTICE] Frames already exist for camera $cam. Skipping ffmpeg extraction."
    continue
  fi

  # --- ffmpegでフレーム抽出 ---
  # -y: 上書き許可
  # -ss: 開始位置へシーク（START_FRAME > 0 の場合のみ）
  # -i: 入力動画
  # -vf "fps=N[,scale=...]": ビデオフィルタ（FPS指定 + オプションのリサイズ）
  # -frames:v N: 最大N枚のフレームを出力
  # -vsync 0: タイムスタンプ変換なし（フレーム落ちを防ぐ）
  # -start_number 0: 出力ファイルのナンバリングを0始まりに（ローカル番号）
  # shellcheck disable=SC2086
  ffmpeg -y \
    $SEEK_OPT \
    -i "$video_path" \
    -vf "$VF_FILTERS" \
    -frames:v "$NUM_FRAMES" \
    -vsync 0 \
    -start_number 0 \
    "$cam_dir/%06d.png"

done

echo "==> Done: extracted frames to $OUTPUT_DIR"
