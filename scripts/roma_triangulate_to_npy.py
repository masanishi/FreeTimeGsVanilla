#!/usr/bin/env python3
"""
RoMa三角測量 → フレームごとのNPYファイル生成スクリプト (FreeTimeGS用)

処理内容:
- COLMAPモデル (sparse/0) からカメラ内部/外部パラメータを読み込む
- 各フレームについて: リファレンスカメラと他カメラ間でRoMaマッチング
- マッチ結果から三角測量で3D点群を生成
- points3d_frame%06d.npy と colors_frame%06d.npy を保存

VRAMリーク対策 (サブプロセス分離方式):
- 各フレームを独立したサブプロセスで実行する
- RoMa/PyTorch/CUDAランタイム内部のメモリリーク
  (autocastキャッシュ、cuBLASワークスペース、DINOv2隠蔽テンソル、
   CUDAコンテキスト成長等) はPythonレベルのgc/empty_cacheでは解放不能
- プロセス終了によりOS/CUDAドライバレベルで全GPUメモリが完全回収される
- これによりフレームが進んでもVRAMが一定に保たれ、OOMを完全に防ぐ
"""

import argparse
import gc  # VRAMリーク対策: GC強制実行用
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

# --- pycolmap APIバージョン検出 ---
# pycolmap の新旧APIに対応するため、importを試行する
try:
    from pycolmap import Reconstruction as PyColmapReconstruction
    PYCOLMAP_API = "new"  # pycolmap >= 0.6 の新API
except ImportError:
    try:
        from pycolmap import SceneManager
        PyColmapReconstruction = None
        PYCOLMAP_API = "old"  # pycolmap 旧API (SceneManager)
    except ImportError:
        SceneManager = None
        PyColmapReconstruction = None
        PYCOLMAP_API = None  # pycolmap 未インストール


def flush_vram(device: str):
    """
    GPUのVRAMキャッシュを強制解放する。
    PyTorchのCUDAアロケータは free されたメモリを再利用のためにキャッシュするが、
    異なるサイズのテンソルが繰り返し確保・解放されるとフラグメンテーションが発生する。
    synchronize() で非同期GPU操作の完了を待ってから、
    gc.collect() でPythonオブジェクトを回収し、empty_cache() でCUDAキャッシュを解放する。
    """
    gc.collect()  # Python GCでテンソル参照を確実に解放
    if device == "cuda":
        torch.cuda.synchronize()  # 非同期GPU操作の完了を待機
        # autocastが作成するFP16重みキャッシュを強制解放
        # RoMaはCUDAで常にautocastが有効化されるため、キャッシュが蓄積する
        if hasattr(torch, 'clear_autocast_cache'):
            torch.clear_autocast_cache()
        torch.cuda.empty_cache()  # CUDAキャッシュアロケータの未使用メモリを解放
    elif device == "mps":
        # MPS (Apple Silicon GPU) の場合もキャッシュ解放
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def preload_roma_weights_cpu():
    """
    RoMaモデルの重みをCPUメモリに一度だけプリロードする。

    フレームごとにmatcherを再作成する際、毎回ディスクI/Oやダウンロードを避けるため、
    重み (state_dict) をCPUメモリにキャッシュしておく。
    戻り値: (roma_weights, dinov2_weights) のタプル。利用不可の場合は (None, None)。
    """
    try:
        import romatch
        from romatch.models.model_zoo import weight_urls
    except ImportError:
        return None, None

    roma_url = None
    if hasattr(romatch, "roma_outdoor"):
        roma_url = weight_urls["romatch"]["outdoor"]
    elif hasattr(romatch, "roma_indoor"):
        roma_url = weight_urls["romatch"]["indoor"]

    dinov2_url = weight_urls.get("dinov2")

    weights = None
    dinov2_weights = None
    if roma_url:
        weights = torch.hub.load_state_dict_from_url(roma_url, map_location="cpu")
    if dinov2_url:
        dinov2_weights = torch.hub.load_state_dict_from_url(dinov2_url, map_location="cpu")

    return weights, dinov2_weights


def load_colmap_cameras(colmap_dir: str):
    """
    COLMAPの sparse reconstruction からカメラパラメータを読み込む。
    戻り値: { カメラ名: {"K": 内部行列, "w2c": world-to-camera行列, "dist": 歪み係数} }
    """
    if PYCOLMAP_API is None:
        raise RuntimeError("pycolmap not found. Install with: pip install pycolmap")

    # COLMAPモデルの読み込み（新旧API対応）
    if PYCOLMAP_API == "new":
        rec = PyColmapReconstruction()
        rec.read(colmap_dir)  # sparse/0 ディレクトリからバイナリモデルを読み込む
        imdata = rec.images  # 画像データ（ポーズ含む）
        cameras_dict = rec.cameras  # カメラ内部パラメータ
    else:
        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        imdata = manager.images
        cameras_dict = manager.cameras

    camera_map = {}
    for k in imdata:
        im = imdata[k]

        # カメラの回転・並進を取得して world-to-camera 変換行列を構築
        if PYCOLMAP_API == "new":
            pose = im.cam_from_world()  # カメラ座標系への変換
            quat = pose.rotation.quat  # クォータニオン [x, y, z, w]
            x, y, z, w = quat
            # クォータニオンから回転行列に変換
            rot = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ])
            trans = np.array(pose.translation).reshape(3, 1)  # 並進ベクトル
        else:
            rot = im.R()  # 回転行列
            trans = im.tvec.reshape(3, 1)  # 並進ベクトル

        # 4x4 world-to-camera 変換行列を構築
        w2c = np.concatenate([np.concatenate([rot, trans], 1), np.array([[0, 0, 0, 1]])], axis=0)

        # カメラ内部パラメータ（焦点距離、主点）を取得
        cam = cameras_dict[im.camera_id]
        if PYCOLMAP_API == "new":
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            model_name = str(cam.model)  # カメラモデル名 (PINHOLE, OPENCV等)
            cam_params = cam.params  # 全パラメータ配列
        else:
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            model_name = str(cam.camera_type)
            cam_params = []

        # カメラモデルに応じた歪み係数を抽出
        if "PINHOLE" in model_name or model_name in ["0", "1"]:
            params = np.zeros(0, dtype=np.float32)  # PINHOLEは歪みなし
        elif "SIMPLE_RADIAL" in model_name or model_name == "2":
            k1 = cam_params[3] if len(cam_params) > 3 else 0.0
            params = np.array([k1, 0.0, 0.0, 0.0], dtype=np.float32)
        elif "RADIAL" in model_name or model_name == "3":
            k1 = cam_params[3] if len(cam_params) > 3 else 0.0
            k2 = cam_params[4] if len(cam_params) > 4 else 0.0
            params = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)
        elif "OPENCV" in model_name or model_name == "4":
            k1 = cam_params[4] if len(cam_params) > 4 else 0.0
            k2 = cam_params[5] if len(cam_params) > 5 else 0.0
            p1 = cam_params[6] if len(cam_params) > 6 else 0.0
            p2 = cam_params[7] if len(cam_params) > 7 else 0.0
            params = np.array([k1, k2, p1, p2], dtype=np.float32)
        else:
            params = np.zeros(0, dtype=np.float32)  # 未知のモデルは歪みなし扱い

        # 3x3 カメラ内部行列
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # カメラ名の抽出（パス構造に応じて親フォルダ名 or ファイル名）
        name = Path(im.name)
        cam_name = name.parent.name if ("/" in im.name or "\\" in im.name) else name.stem

        camera_map[cam_name] = {
            "K": K,       # カメラ内部行列 [3, 3]
            "w2c": w2c,   # world-to-camera 変換行列 [4, 4]
            "dist": params,  # 歪み係数
        }

    return camera_map


def try_init_roma(device, amp, upsample_preds=True,
                   preloaded_weights=None, preloaded_dinov2_weights=None,
                   quiet=False):
    """
    RoMaマッチャーを初期化する。

    upsample_preds: Trueの場合、高解像度(864x864)のアップサンプリングpassを追加実行する。
                    Falseの場合、coarse解像度(560x560)のみでVRAMを大幅に削減できる。
    preloaded_weights: CPUにプリロード済みのRoMa重み (state_dict)。
                       Noneの場合はtorch.hubから自動ダウンロード。
    preloaded_dinov2_weights: CPUにプリロード済みのDINOv2重み (state_dict)。
    quiet: Trueの場合、デバッグ情報の出力を抑制する（フレームループ内での繰り返し出力防止）。
    """
    try:
        import romatch
    except Exception:
        return None  # romatch未インストール

    # デバッグ情報の表示（初回のみ）
    if not quiet:
        print(f"[ROMA][DEBUG] romatch module path: {getattr(romatch, '__file__', 'N/A')}")
        print(f"[ROMA][DEBUG] available attrs: {', '.join([k for k in ['roma_outdoor', 'roma_indoor', 'tiny_roma_v1_outdoor'] if hasattr(romatch, k)])}")
        print(f"[ROMA][DEBUG] upsample_preds={upsample_preds}")

    # AMP (Automatic Mixed Precision) の精度設定
    amp_dtype = torch.float16 if amp else torch.float32

    # 利用可能なRoMaモデルを優先度順に試行
    if hasattr(romatch, "roma_outdoor"):
        # roma_outdoor: 屋外シーン向け高精度モデル（DINOv2 ViT-L ベース）
        matcher = romatch.roma_outdoor(
            device=device,
            amp_dtype=amp_dtype,
            symmetric=False,
            upsample_preds=upsample_preds,
            weights=preloaded_weights,
            dinov2_weights=preloaded_dinov2_weights,
        )
    elif hasattr(romatch, "roma_indoor"):
        # roma_indoor: 屋内シーン向けモデル
        matcher = romatch.roma_indoor(
            device=device,
            amp_dtype=amp_dtype,
            symmetric=False,
            upsample_preds=upsample_preds,
            weights=preloaded_weights,
            dinov2_weights=preloaded_dinov2_weights,
        )
    elif hasattr(romatch, "tiny_roma_v1_outdoor"):
        # tiny_roma: 軽量モデル（DINOv2不使用、upsample_predsパラメータなし）
        matcher = romatch.tiny_roma_v1_outdoor(device=device)
    else:
        return None  # 利用可能なモデルなし

    # 推論モードに設定（ドロップアウト等を無効化）
    if hasattr(matcher, "eval"):
        matcher.eval()
    return matcher


def match_roma(matcher, img0, img1, cert_th=0.3, max_matches=20000):
    """
    RoMaマッチャーで2枚の画像間の対応点を検出する。

    RoMaは正規化座標 [-1, 1] で対応を出力する。
    ピクセル座標への変換は入力画像の元サイズを使用する:
      pixel_x = W / 2 * (norm_x + 1)
    ※ warp mapの解像度（内部upsample_res 864x1152等）ではない点に注意
    """
    # 入力画像のサイズを取得（座標変換に使用）
    W_A, H_A = img0.size  # PIL Image.size は (width, height) を返す
    W_B, H_B = img1.size

    # 推論モードでマッチング実行（勾配計算を無効化してVRAM節約）
    with torch.inference_mode():
        warp_t, certainty_t = matcher.match(img0, img1, batched=True)

        # バッチ次元を除去（batched=Trueの場合 [1, H, W, 4] → [H, W, 4]）
        # inference_mode コンテキスト内で CPU 転送まで完了させる
        if warp_t.ndim == 4:
            warp_t = warp_t[0]
        if certainty_t.ndim == 3:
            certainty_t = certainty_t[0]

        # GPU → CPU に転送してnumpy配列に変換（GPUテンソルを即座に解放するため）
        warp = warp_t.detach().cpu().numpy()
        certainty = certainty_t.detach().cpu().numpy()
        del warp_t, certainty_t  # GPUテンソルの参照を明示的に解放

    # warp map を平坦化: [H, W, 4] → [H*W, 4]  (4 = [ax, ay, bx, by] の正規化座標)
    h, w, _ = warp.shape
    flat_warp = warp.reshape(-1, 4)
    flat_cert = certainty.reshape(-1)

    # 確信度フィルタ: cert_th 以上のマッチのみ保持
    keep = flat_cert >= cert_th
    if keep.sum() == 0:
        # マッチなし → 空配列を返す
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    pts = flat_warp[keep]
    scores = flat_cert[keep]

    # マッチ数が上限を超えた場合、ランダムサンプリングで制限
    if len(scores) > max_matches:
        idx = np.random.choice(len(scores), max_matches, replace=False)
        pts = pts[idx]
        scores = scores[idx]

    # 正規化座標 [-1, 1] → ピクセル座標に変換
    # (RoMaの to_pixel_coordinates / warp_to_pixel_coords に準拠)
    ax = W_A / 2 * (pts[:, 0] + 1)  # 画像Aの x ピクセル座標
    ay = H_A / 2 * (pts[:, 1] + 1)  # 画像Aの y ピクセル座標
    bx = W_B / 2 * (pts[:, 2] + 1)  # 画像Bの x ピクセル座標
    by = H_B / 2 * (pts[:, 3] + 1)  # 画像Bの y ピクセル座標

    pts0 = np.stack([ax, ay], axis=1).astype(np.float32)  # 画像Aの対応点 [M, 2]
    pts1 = np.stack([bx, by], axis=1).astype(np.float32)  # 画像Bの対応点 [M, 2]
    scores = scores.astype(np.float32)  # 確信度スコア [M]
    return pts0, pts1, scores


def triangulate_pair(K0, w2c0, K1, w2c1, pts0, pts1, dist0=None, dist1=None):
    """
    2つのカメラビューの対応点からDLT三角測量で3D点を復元する。

    歪み係数 (dist0, dist1) が渡された場合:
      1. cv2.undistortPoints() でピクセル座標 → 歪み補正済み正規化座標に変換
      2. 射影行列は P = [R | t]（K不要、正規化座標での三角測量）
    歪み係数がない場合:
      従来通り P = K @ [R | t] でピクセル座標のまま三角測量
    """
    has_dist0 = dist0 is not None and len(dist0) > 0
    has_dist1 = dist1 is not None and len(dist1) > 0

    if has_dist0 or has_dist1:
        # 歪み補正: ピクセル座標 → 歪み補正済み正規化座標
        # cv2.undistortPoints(src, K, dist) → 正規化座標 [M, 1, 2]
        d0 = dist0 if has_dist0 else np.zeros(4, dtype=np.float32)
        d1 = dist1 if has_dist1 else np.zeros(4, dtype=np.float32)
        pts0_undist = cv2.undistortPoints(pts0.reshape(-1, 1, 2), K0, d0)  # [M, 1, 2]
        pts1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), K1, d1)  # [M, 1, 2]
        pts0_h = pts0_undist.reshape(-1, 2).T  # [2, M]
        pts1_h = pts1_undist.reshape(-1, 2).T  # [2, M]
        # 正規化座標での三角測量: P = [R | t]（Kは undistortPoints 内で既に適用済み）
        P0 = w2c0[:3, :]  # [3, 4]
        P1 = w2c1[:3, :]  # [3, 4]
    else:
        # 歪みなし: 従来通りピクセル座標で三角測量
        P0 = K0 @ w2c0[:3, :]  # カメラ0の射影行列 [3, 4]
        P1 = K1 @ w2c1[:3, :]  # カメラ1の射影行列 [3, 4]
        pts0_h = pts0.T  # [2, M] に転置（OpenCV形式）
        pts1_h = pts1.T  # [2, M] に転置

    # DLT三角測量: 同次座標 [4, M] を取得
    X_h = cv2.triangulatePoints(P0, P1, pts0_h, pts1_h)
    # 同次座標 → 3D座標に変換: [4, M] → [M, 3]
    X = (X_h[:3] / X_h[3:4]).T
    return X


def sample_colors(img, pts):
    """
    画像から指定座標のRGB色をサンプリングする。
    img: [H, W, 3] RGB画像 (0-255)
    pts: [M, 2] ピクセル座標 (x, y)
    """
    h, w = img.shape[:2]
    # 座標を画像範囲内にクリップ（四捨五入して最近傍ピクセルを取得）
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return img[ys, xs, :].astype(np.float32)  # [M, 3] RGB色 (0-255)


def format_memory_stats(device: str) -> str:
    """
    GPU VRAM使用状況をフォーマットして返す。
    allocated: 現在テンソルに使用中のVRAM
    reserved: PyTorchがキャッシュとして確保しているVRAM（allocatedより大きい）
    peak: プロセス開始 or リセット以降のピークallocated
    """
    if device == "cuda":
        try:
            allocated = torch.cuda.memory_allocated()  # 現在使用中のVRAM
            reserved = torch.cuda.memory_reserved()  # キャッシュとして確保中のVRAM
            peak = torch.cuda.max_memory_allocated()  # ピーク使用量
            return (
                f"CUDA allocated={allocated / (1024 ** 2):.1f}MB, "
                f"reserved={reserved / (1024 ** 2):.1f}MB, "
                f"peak={peak / (1024 ** 2):.1f}MB"
            )
        except Exception:
            return "CUDA memory: N/A"
    if device == "mps":
        try:
            allocated = torch.mps.current_allocated_memory()
            reserved = torch.mps.current_reserved_memory()
            return f"MPS allocated={allocated / (1024 ** 2):.1f}MB, reserved={reserved / (1024 ** 2):.1f}MB"
        except Exception:
            return "MPS memory: N/A"
    return "Device memory: N/A"


def main():
    # --- コマンドライン引数の定義 ---
    parser = argparse.ArgumentParser(
        description="RoMa三角測量でフレームごとの3D点群NPYを生成する"
    )
    parser.add_argument("--images-dir", required=True,
                        help="カメラ別画像ディレクトリ (images/CCCC/FFFFFF.png)")
    parser.add_argument("--colmap-model", required=True,
                        help="COLMAPモデルディレクトリ (sparse/0)")
    parser.add_argument("--output-dir", required=True,
                        help="NPY出力ディレクトリ")
    parser.add_argument("--frame-start", type=int, default=0,
                        help="開始フレーム番号")
    parser.add_argument("--frame-end", type=int, default=59,
                        help="終了フレーム番号")
    parser.add_argument("--frame-step", type=int, default=1,
                        help="フレーム間隔（1=全フレーム処理）")
    parser.add_argument("--ref-cam", default="0000",
                        help="リファレンスカメラID（このカメラと他カメラでペアマッチする）")
    parser.add_argument("--image-ext", default="png",
                        help="画像ファイルの拡張子")
    parser.add_argument("--matcher", choices=["roma"], default="roma",
                        help="マッチングアルゴリズム（現在はromaのみ）")
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto",
                        help="使用デバイス（auto=CUDA優先、なければMPS）")
    parser.add_argument("--certainty", type=float, default=0.3,
                        help="RoMa確信度しきい値（0.0-1.0、低いほどマッチが多い）")
    parser.add_argument("--use-ransac", action="store_true",
                        help="RANSACで外れ値マッチを除去する")
    parser.add_argument("--ransac-th", type=float, default=0.5,
                        help="RANSACのピクセル誤差しきい値")
    parser.add_argument("--min-depth", type=float, default=1e-4,
                        help="三角測量後の最小深度フィルタ（カメラ背後の点を除去）")
    parser.add_argument("--max-matches", type=int, default=20000,
                        help="ペアあたりの最大マッチ数（ランダムサンプリング）")
    parser.add_argument("--voxel-size", type=float, default=0.0,
                        help="ボクセルダウンサンプリングサイズ（0=無効）")
    parser.add_argument("--image-scale", type=float, default=1.0,
                        help="RoMa入力画像のスケーリング係数（1.0=原寸）")
    parser.add_argument("--amp", action="store_true",
                        help="AMP (FP16混合精度) を有効にしてVRAM使用量を削減する")
    parser.add_argument("--no-upsample", action="store_true",
                        help="RoMaのアップサンプリングpassを無効化する "
                             "(解像度 864→560 に下がるがVRAMを大幅に削減)")
    parser.add_argument("--cache-dir", default="",
                        help="モデルキャッシュディレクトリ（未使用）")
    # 内部用: サブプロセスワーカーモードフラグ（ユーザーは使用しない）
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ==================================================================
    # サブプロセス分離モード (MASTER / WORKER)
    # ==================================================================
    # RoMa + PyTorch + CUDA ランタイムは、match() 呼び出しごとに
    # Python GC では回収不能な GPU メモリを蓄積する:
    #   - torch.autocast の FP16 重みキャッシュ (clear_autocast_cache で部分的に解放)
    #   - cuBLAS / cuDNN ワークスペース (CUDA コンテキスト破棄でのみ解放)
    #   - DINOv2 ViT の内部バッファ (plain list で隠蔽されており nn.Module.to() で管理不可)
    #   - CUDA コンテキスト自体の成長 (ドライバレベルのメモリプール)
    # これらは gc.collect() + empty_cache() + clear_autocast_cache() でも
    # 完全には解放できず、フレームが進むごとに VRAM 使用量が単調増加し OOM に至る。
    #
    # 唯一の確実な解決策: プロセスを終了させて OS / CUDA ドライバに
    # 全 GPU メモリを回収させること。
    #
    # 動作:
    #   --_worker なし (通常起動 = MASTER):各フレームで自身をサブプロセスとして起動
    #   --_worker あり (WORKER): 指定フレームを処理して exit (GPU メモリ完全解放)
    # ==================================================================

    if not args._worker:
        # ====== MASTER MODE: フレームごとにサブプロセスを起動 ======
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 基本バリデーション（GPU を使わずに実行）
        if not os.path.isdir(args.colmap_model):
            raise FileNotFoundError(f"COLMAP model directory not found: {args.colmap_model}")
        if not os.path.isdir(args.images_dir):
            raise FileNotFoundError(f"Images directory not found: {args.images_dir}")

        frame_range = list(range(args.frame_start, args.frame_end + 1, args.frame_step))
        total = len(frame_range)
        print(f"[MASTER] Subprocess-per-frame mode for complete VRAM isolation")
        print(f"[MASTER] Frames: {args.frame_start} → {args.frame_end} (step={args.frame_step}, total={total})")

        failed_frames = []
        for i, frame_idx in enumerate(frame_range):
            print(f"\n{'='*60}")
            print(f"[MASTER] Frame {frame_idx:06d}  ({i+1}/{total})")
            print(f"{'='*60}")

            # 自身を --_worker モードで起動するコマンドを構築
            cmd = [
                sys.executable, os.path.abspath(__file__),
                '--images-dir', str(args.images_dir),
                '--colmap-model', str(args.colmap_model),
                '--output-dir', str(args.output_dir),
                '--frame-start', str(frame_idx),
                '--frame-end', str(frame_idx),
                '--frame-step', '1',
                '--ref-cam', args.ref_cam,
                '--image-ext', args.image_ext,
                '--matcher', args.matcher,
                '--device', args.device,
                '--certainty', str(args.certainty),
                '--max-matches', str(args.max_matches),
                '--min-depth', str(args.min_depth),
                '--voxel-size', str(args.voxel_size),
                '--image-scale', str(args.image_scale),
                '--ransac-th', str(args.ransac_th),
            ]
            if args.use_ransac:
                cmd.append('--use-ransac')
            if args.amp:
                cmd.append('--amp')
            if args.no_upsample:
                cmd.append('--no-upsample')
            if args.cache_dir:
                cmd += ['--cache-dir', args.cache_dir]
            cmd.append('--_worker')

            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"[MASTER][ERROR] Frame {frame_idx:06d} failed (exit code: {result.returncode})")
                failed_frames.append(frame_idx)

        print(f"\n{'='*60}")
        if failed_frames:
            print(f"[MASTER] Completed with {len(failed_frames)} failed frame(s): {failed_frames}")
        else:
            print(f"[MASTER] All {total} frames completed successfully.")
        return

    # ====== WORKER MODE: 単一フレーム（またはフレーム範囲）を処理 ======

    # --- パスの準備 ---
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # 出力ディレクトリを作成

    # --- COLMAPカメラパラメータの読み込み ---
    camera_map = load_colmap_cameras(args.colmap_model)
    camera_names = sorted(camera_map.keys())  # カメラ名をソートして順序を固定

    # リファレンスカメラの存在チェック
    if args.ref_cam not in camera_map:
        raise ValueError(f"ref cam not found in COLMAP: {args.ref_cam}")

    # --- デバイス選択 ---
    if args.device == "auto":
        # CUDA > MPS の優先順でデバイスを自動選択
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            raise RuntimeError("Neither CUDA nor MPS is available. Enable GPU/MPS or set up a supported device.")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but not available. Set --device mps or enable CUDA.")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but not available. Set --device cuda or enable MPS.")

    # MPS使用時のCPUフォールバック設定（一部のopsがMPS未対応のため）
    if args.device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        print("[NOTICE] Setting PYTORCH_ENABLE_MPS_FALLBACK=1 to enable CPU fallback for unsupported MPS ops.")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # --- RoMaモデル重みのプリロード ---
    # VRAMリーク対策: フレームごとにmatcherを再作成するため、
    # 重みをCPUメモリにキャッシュしておく（ディスクI/O・ダウンロードの回避）
    upsample_preds = not args.no_upsample
    print("[ROMA] Pre-loading model weights to CPU...")
    roma_weights_cpu, dinov2_weights_cpu = preload_roma_weights_cpu()

    # 初回のmatcher作成（利用可能性チェック + デバッグ情報表示用）
    roma_matcher = try_init_roma(
        args.device, args.amp, upsample_preds=upsample_preds,
        preloaded_weights=roma_weights_cpu,
        preloaded_dinov2_weights=dinov2_weights_cpu,
        quiet=False,
    )
    if roma_matcher is None:
        raise RuntimeError("ROMA is not available. Install romatch and ensure GPU/MPS support is enabled.")
    print(f"[ROMA] matcher initialized on {args.device} (upsample_preds={upsample_preds})")
    # 初回matcherは即座に解放（フレームループ内で再作成する）
    del roma_matcher
    flush_vram(args.device)

    # --- フレームごとの三角測量ループ ---
    for frame_idx in tqdm(range(args.frame_start, args.frame_end + 1, args.frame_step), desc="Triangulating frames"):

        # CUDAの場合、フレーム開始時にピークメモリ統計をリセット
        # （フレーム単位でのピークVRAM追跡用）
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # VRAMリーク対策: フレームごとにmatcherを新規作成する。
        # RoMaのmatch()呼び出しで蓄積される内部GPUテンソル（autocastキャッシュ、
        # 特徴マップ、DINOv2内部状態等）を確実に解放するため、フレーム終了時に
        # matcherオブジェクトを破棄してVRAMを完全にクリーンにする。
        # CPUにプリロードした重みから再作成するため、追加のディスクI/Oは発生しない。
        roma_matcher = try_init_roma(
            args.device, args.amp, upsample_preds=upsample_preds,
            preloaded_weights=roma_weights_cpu,
            preloaded_dinov2_weights=dinov2_weights_cpu,
            quiet=True,
        )
        if roma_matcher is None:
            raise RuntimeError("ROMA matcher creation failed.")

        # リファレンスカメラの画像パスを構築
        ref_path = images_dir / args.ref_cam / f"{frame_idx:06d}.{args.image_ext}"
        if not ref_path.exists():
            print(f"[WARN] missing ref frame: {ref_path}")
            continue

        # リファレンス画像を読み込み（BGRで読み込まれる）
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            print(f"[WARN] failed to read: {ref_path}")
            continue

        # 色サンプリング用にRGBに変換して保持
        # （downstream の combine_frames / trainer はRGBを期待する）
        ref_img_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

        # RoMa入力用にスケーリング（image_scale != 1.0 の場合のみ）
        if args.image_scale != 1.0:
            ref_img = cv2.resize(ref_img, dsize=None, fx=args.image_scale, fy=args.image_scale)

        all_points = []  # このフレームの全3D点を集約するリスト
        all_colors = []  # このフレームの全RGB色を集約するリスト

        # ref_pil をフレームごとに1回だけ生成（カメラペアループの外）
        # ref_img は既にスケーリング済み（image_scale != 1.0 の場合）
        ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))

        # --- カメラペアループ: refカメラ × 他の全カメラ ---
        for cam_name in camera_names:
            # リファレンスカメラ自身はスキップ
            if cam_name == args.ref_cam:
                continue

            # 他カメラの同フレーム画像パスを構築
            other_path = images_dir / cam_name / f"{frame_idx:06d}.{args.image_ext}"
            if not other_path.exists():
                continue

            # 他カメラの画像を読み込み
            other_img = cv2.imread(str(other_path))
            if other_img is None:
                continue

            # RoMa入力用にスケーリング
            if args.image_scale != 1.0:
                other_img = cv2.resize(other_img, dsize=None, fx=args.image_scale, fy=args.image_scale)

            # OpenCV BGR → PIL RGB に変換（RoMaの入力形式）
            other_pil = Image.fromarray(cv2.cvtColor(other_img, cv2.COLOR_BGR2RGB))

            # RoMaマッチング: 2画像間の対応点を検出
            pts0, pts1, scores = match_roma(
                roma_matcher,
                ref_pil,
                other_pil,
                cert_th=args.certainty,
                max_matches=args.max_matches,
            )

            # 確信度でフィルタリング（match_roma内部でも行うが二重チェック）
            if len(scores) > 0:
                keep = scores >= args.certainty
                pts0, pts1, scores = pts0[keep], pts1[keep], scores[keep]

            # マッチ数が最低限（8点: 基礎行列推定に必要）に満たない場合はスキップ
            if len(pts0) < 8:
                # VRAMリーク対策: スキップ前にもGPUキャッシュを解放
                del other_img, other_pil
                flush_vram(args.device)
                continue

            # RANSAC外れ値除去（オプション）
            if args.use_ransac:
                # 基礎行列FをRANSACで推定し、インライアのみ残す
                F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, args.ransac_th, 0.999)
                if mask is not None:
                    mask = mask.squeeze().astype(bool)
                    pts0 = pts0[mask]
                    pts1 = pts1[mask]

            # RANSAC後もマッチ数チェック
            if len(pts0) < 8:
                del other_img, other_pil
                flush_vram(args.device)
                continue

            # 三角測量にはスケーリング前の座標が必要なので元に戻す
            if args.image_scale != 1.0:
                pts0 = pts0 / args.image_scale
                pts1 = pts1 / args.image_scale

            # カメラパラメータを取得
            K0 = camera_map[args.ref_cam]["K"]    # refカメラの内部行列
            w2c0 = camera_map[args.ref_cam]["w2c"]  # refカメラのw2c行列
            dist0 = camera_map[args.ref_cam]["dist"]  # refカメラの歪み係数
            K1 = camera_map[cam_name]["K"]         # 他カメラの内部行列
            w2c1 = camera_map[cam_name]["w2c"]     # 他カメラのw2c行列
            dist1 = camera_map[cam_name]["dist"]   # 他カメラの歪み係数

            # DLT三角測量: 対応点から3D点群を復元（歪み補正付き）
            X = triangulate_pair(K0, w2c0, K1, w2c1, pts0, pts1, dist0, dist1)

            # 深度フィルタ: 両カメラから正の深度にある点のみ保持
            X_h = np.concatenate([X, np.ones((len(X), 1))], axis=1)  # 同次座標 [M, 4]
            depth0 = (w2c0 @ X_h.T).T[:, 2]  # refカメラからの深度
            depth1 = (w2c1 @ X_h.T).T[:, 2]  # 他カメラからの深度
            valid = (depth0 > args.min_depth) & (depth1 > args.min_depth)
            X = X[valid]  # 有効な3D点のみ保持
            pts0 = pts0[valid]  # 対応する2D座標も保持（色サンプリング用）

            if len(X) == 0:
                # 有効点なし → スキップ
                del other_img, other_pil
                flush_vram(args.device)
                continue

            # 元スケールのRGB画像から色をサンプリング（pts0は元スケールの座標）
            colors = sample_colors(ref_img_rgb, pts0)
            all_points.append(X)      # 3D点群を集約リストに追加
            all_colors.append(colors)  # RGB色を集約リストに追加

            # このカメラペアで使用した変数を明示的に解放
            del other_img, other_pil, pts0, pts1, scores, X, colors

            # VRAMリーク対策: 各カメラペア処理後にGPUキャッシュを強制解放
            # PyTorchのCUDAアロケータはfreeされたメモリを再利用のためにキャッシュするが、
            # 異なるサイズのテンソルの繰り返し確保・解放でフラグメンテーションが発生する。
            # 毎ペアで flush することでフラグメンテーションの蓄積を防ぐ。
            flush_vram(args.device)

        # --- フレームの全ペア処理完了 ---

        # マッチが1つもなかった場合
        if len(all_points) == 0:
            print(f"[WARN] no points for frame {frame_idx}")
            continue

        # 全カメラペアの3D点群・色を結合
        points = np.concatenate(all_points, axis=0).astype(np.float32)  # [N, 3]
        colors = np.concatenate(all_colors, axis=0).astype(np.float32)  # [N, 3]

        # ボクセルダウンサンプリング（オプション: voxel_size > 0 の場合）
        if args.voxel_size and args.voxel_size > 0:
            # 空間ハッシュで同一ボクセル内の点を1つに統合
            vox = np.floor(points / args.voxel_size).astype(np.int64)
            key = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
            _, unique_idx = np.unique(key, return_index=True)
            points = points[unique_idx]
            colors = colors[unique_idx]

        # NPYファイルに保存
        np.save(output_dir / f"points3d_frame{frame_idx:06d}.npy", points)  # 3D座標
        np.save(output_dir / f"colors_frame{frame_idx:06d}.npy", colors)    # RGB色

        # フレーム完了ログ（VRAM統計付き）
        print(f"[Frame {frame_idx:06d}] points={len(points)} | {format_memory_stats(args.device)}")

        # このフレームで使用した変数を明示的に解放
        del points, colors, all_points, all_colors, ref_img, ref_img_rgb, ref_pil

        # VRAMリーク対策（核心部分）: matcherオブジェクトを破棄してGPUメモリを完全解放。
        # RoMaのmatch()は内部でautocast FP16キャッシュ、CUDAワークスペース、
        # DINOv2の隠蔽されたGPUテンソル等を蓄積する。これらはgc.collect() +
        # empty_cache()だけでは解放できないため、matcherオブジェクト自体を破棄し、
        # 次フレームで新規作成することでVRAMを確実にクリーンな状態に戻す。
        del roma_matcher
        flush_vram(args.device)


if __name__ == "__main__":
    main()
