# HOTWO データセット準備（Ubuntu / VastAI CUDA 12.8 Docker 前提）

この手順は **`XXXX.mp4`（多視点同期動画）から学習開始まで**を一気通貫でまとめたものです。  
対象リポジトリ: `FreeTimeGsVanilla`

---

## 0. 前提

- Ubuntu Linux（VastAI）
- CUDA 12.8 の Docker image（GPU利用）
- `git`, `cmake`, `ffmpeg`, `python3` が使えること

---

## 1. リポジトリのクローン

```bash
git clone https://github.com/your_org/FreeTimeGsVanilla.git
cd FreeTimeGsVanilla
```

---

## 2. Python 環境（uv で統一）

```bash
sudo apt-get update
sudo apt-get install -y python3

uv venv .venv
source .venv/bin/activate

uv pip install --upgrade pip
```

---

## 2.5. pycolmap（CUDA12 対応）をインストール

pycolmap の CUDA12 対応版は **パッケージ名が違っても import 名は `pycolmap`** です。

```bash
# CUDA12 対応ビルドをインストール
uv pip install pycolmap-cuda12
```

### 依存ライブラリ（Ubuntu）

pycolmap のバックエンドが読み込めない場合は X11 依存が不足している可能性があります。

```bash
sudo apt-get install -y libsm6 libxext6 libxrender1 libice6
```

---

## 3. COLMAP（CLI も選択可）

基本は `pycolmap` を使いますが、**CUDA対応の `colmap` CLI** でも再構成できます。

---

## 4. RoMa（romatch）のインストール

### PyTorch（CUDA）

※ CUDA 12.8 でも **cu121** ホイールで動くケースが多いです。

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### RoMa + 依存

```bash
uv pip install "romatch[fused-local-corr]" opencv-python imageio pillow numpy
```

---

## 5. MP4 → 画像フレーム抽出（scripts で実行）

Step 5 は `scripts/prepare_dataset.sh` で実行できます。

```bash
./scripts/prepare_dataset.sh <videos_dir> <data_dir> <frame_start> <frame_end> [fps] [downsample] [skip_extract]
```

前提:
- `<videos_dir>` に `cam01.mp4, cam02.mp4, ...` のようなカメラ名付き動画がある

出力:
- `<data_dir>/images/cam01_frame000000.png` のように **フラット構成**で展開されます

---

## 6. 画像配置（scripts で確認）

Step 6 も `scripts/prepare_dataset.sh` の出力で満たせます。  
`dataset/images` は **フラット構成**のまま進めます。

```
dataset/images/
├── cam01_frame000000.png
├── cam01_frame000001.png
├── cam02_frame000000.png
└── ...
```

---

## 7. COLMAP 再構成（pycolmap）

CLI の `colmap` は使わず、`pycolmap` で再構成します。

```bash
python scripts/prepare_dataset.py ./dance ./dataset 0 60 60 2 true --use-gpu 1
```

結果は `dataset/sparse/0/` に出力されます。

---

## 7.1 COLMAP 再構成（CLI / CUDA 対応版）

CUDA 対応の `colmap` を使う場合のコマンド例です。

```bash
colmap database_creator --database_path dataset/database.db

colmap feature_extractor \
  --database_path dataset/database.db \
  --image_path dataset/images \
  --ImageReader.single_camera 1 \
  --FeatureExtraction.use_gpu 1

colmap exhaustive_matcher \
  --database_path dataset/database.db \
  --FeatureMatching.use_gpu 1

colmap mapper \
  --database_path dataset/database.db \
  --image_path dataset/images \
  --output_path dataset/sparse \
  --Mapper.ba_use_gpu 1
```

結果は `dataset/sparse/0/` に出力されます。

---

## 8. RoMa 三角測量 → NPY 出力

```bash
python scripts/roma_triangulate_to_npy.py \
  --data-dir ./dataset \
  --out-dir ./triangulation_output \
  --frame-start 0 \
  --frame-end 5 \
  --ref-cam 0000 \
  --use-ransac
```

出力:
```
triangulation_output/
├── points3d_frame000000.npy
├── colors_frame000000.npy
└── ...
```

---

## 9. NPY → NPZ に結合

```bash
python src/combine_frames_fast_keyframes.py \
  --input-dir ./triangulation_output \
  --output-path ./keyframes.npz \
  --frame-start 0 \
  --frame-end 5 \
  --keyframe-step 5
```

---

## 10. 学習開始

```bash
CUDA_VISIBLE_DEVICES=0 python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe_small \
  --data-dir ./dataset \
  --init-npz-path ./keyframes.npz \
  --result-dir ./results \
  --start-frame 0 \
  --end-frame 61
```

---

## 補足: scripts/ のデータセット準備スクリプト

`scripts` フォルダのスクリプトは **データセット準備用途**です。README の入力要件（`images/` はフラット構成）と整合する使い方は以下です。

### 一括で画像抽出 + COLMAP まで（pycolmap）

`scripts/prepare_dataset.sh` は **手順 5〜7 をまとめて実行**します（CLI の `colmap` は使いません）。

```bash
./scripts/prepare_dataset.sh <videos_dir> <data_dir> <frame_start> <frame_end> [fps] [downsample] [skip_extract]
```

前提:
- `<videos_dir>` に `cam01.mp4, cam02.mp4, ...` のようなカメラ名付き動画があること
- フレームは `data_dir/images/cam01_frame000000.png` のように **フラット**に展開されます

### 画像をカメラ別フォルダに変換（基本は不要）

`scripts/convert_images_to_cam_folders.sh` は **フラット → ネスト**変換です。  
README の入力要件とは異なるため、この手順は **通常不要**です。

### RoMa での三角測量

`scripts/roma_triangulate_to_npy.py` は **フラット/ネストどちらも対応**ですが、README に合わせるなら **フラット構成のまま実行**してください。

---

## 参考

- RoMa: https://github.com/Parskatt/RoMa
- COLMAP: https://github.com/colmap/colmap
