#!/usr/bin/env python3
"""
Quick .pt → simple-viewer binary converter (no filtering, no gsplat dependency).

再現度確認のため、チェックポイントの全Gaussianをフィルタリングなしで
simple-viewerで表示できるバイナリフォーマット (.ftgs) に変換する。

Outputs:
  checkpoint.ftgs   — 単一バイナリ (全スプラットデータ)
  scene_meta.json   — 最小限のメタデータ (center, radius, totalFrames)

Usage:
    # 基本 (simple-viewer/public/data/ に出力, totalFrames=60):
    python scripts/pt_to_viewer.py ckpt_49999.pt

    # 出力先・フレーム数を指定:
    python scripts/pt_to_viewer.py ckpt_49999.pt \\
        --output-dir simple-viewer/public/data --total-frames 60

    # COLMAPカメラプリセット付き:
    python scripts/pt_to_viewer.py ckpt_49999.pt \\
        --colmap-dir dataset/dance/colmap/sparse/0

Binary format (.ftgs):
    Header (32 bytes, little-endian):
      uint32 magic            = 0x46544753 ("FTGS")
      uint32 version          = 1
      uint32 numSplats        = N
      uint32 totalFrames
      uint32 shDegree         (0-3)
      uint32 shCoeffsPerChan  (K: 0, 3, 8, or 15)
      uint32[2] reserved

    Data (contiguous float32):
      float32[N*3]      means        (x, y, z)
      float32[N*3]      scales       (log-space)
      float32[N*4]      quats        (w, x, y, z)
      float32[N]        opacities    (logit-space)
      float32[N*3]      sh0          (DC: r, g, b)
      float32[N*K*3]    shN          ([N, K, 3] C-order)
      float32[N*3]      velocities   (vx, vy, vz)
      float32[N]        times        (mu_t)
      float32[N]        durations    (log-space)
"""

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path

import numpy as np
import torch


MAGIC = 0x46544753  # "FTGS" in little-endian
VERSION = 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert .pt checkpoint to simple-viewer binary (.ftgs)"
    )
    parser.add_argument("ckpt", type=str, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: simple-viewer/public/data)",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=60,
        help="Total frames in sequence (default: 60)",
    )
    parser.add_argument(
        "--colmap-dir",
        type=str,
        default=None,
        help="COLMAP sparse model dir for camera presets",
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir is None:
        project_root = Path(__file__).resolve().parent.parent
        args.output_dir = str(project_root / "simple-viewer" / "public" / "data")

    os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load checkpoint ───
    print(f"Loading {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    means = splats["means"].cpu().numpy().astype(np.float32)           # [N, 3]
    scales = splats["scales"].cpu().numpy().astype(np.float32)         # [N, 3] log
    quats = splats["quats"].cpu().numpy().astype(np.float32)           # [N, 4]
    opacities = splats["opacities"].cpu().numpy().astype(np.float32)   # [N, 1] or [N] logit
    sh0 = splats["sh0"].cpu().numpy().astype(np.float32)               # [N, 1, 3]
    shN = splats["shN"].cpu().numpy().astype(np.float32)               # [N, K, 3]
    velocities = splats["velocities"].cpu().numpy().astype(np.float32) # [N, 3]
    times = splats["times"].cpu().numpy().astype(np.float32)           # [N, 1] or [N]
    durations = splats["durations"].cpu().numpy().astype(np.float32)   # [N, 1] or [N] log

    N = means.shape[0]
    K = shN.shape[1] if shN.ndim == 3 else 0

    # Determine SH degree from K
    # K = (deg+1)^2 - 1: 0→0, 3→1, 8→2, 15→3
    sh_degree_map = {0: 0, 3: 1, 8: 2, 15: 3}
    sh_degree = sh_degree_map.get(K, 3)

    # Flatten arrays to ensure contiguous C-order
    means_flat = means.reshape(-1)                    # [N*3]
    scales_flat = scales.reshape(-1)                  # [N*3]
    quats_flat = quats.reshape(-1)                    # [N*4]
    opacities_flat = opacities.reshape(-1)            # [N]
    sh0_flat = sh0.reshape(N, 3).reshape(-1)          # [N*3]  (squeeze the middle dim)
    shN_flat = shN.reshape(-1) if K > 0 else np.array([], dtype=np.float32)  # [N*K*3]
    velocities_flat = velocities.reshape(-1)          # [N*3]
    times_flat = times.reshape(-1)                    # [N]
    durations_flat = durations.reshape(-1)            # [N]

    print(f"  Splats: {N:,}")
    print(f"  SH degree: {sh_degree} (K={K})")
    print(f"  Means range: [{means.min():.3f}, {means.max():.3f}]")
    print(f"  Velocity mag: [{np.linalg.norm(velocities, axis=1).min():.4f}, "
          f"{np.linalg.norm(velocities, axis=1).max():.4f}]")
    print(f"  Times: [{times_flat.min():.4f}, {times_flat.max():.4f}]")

    # ─── Write .ftgs binary ───
    ftgs_path = os.path.join(args.output_dir, "checkpoint.ftgs")
    header = struct.pack(
        "<8I",
        MAGIC,
        VERSION,
        N,
        args.total_frames,
        sh_degree,
        K,
        0,  # reserved
        0,  # reserved
    )
    assert len(header) == 32

    with open(ftgs_path, "wb") as f:
        f.write(header)
        f.write(means_flat.tobytes())
        f.write(scales_flat.tobytes())
        f.write(quats_flat.tobytes())
        f.write(opacities_flat.tobytes())
        f.write(sh0_flat.tobytes())
        f.write(shN_flat.tobytes())
        f.write(velocities_flat.tobytes())
        f.write(times_flat.tobytes())
        f.write(durations_flat.tobytes())

    size_mb = os.path.getsize(ftgs_path) / (1024 * 1024)
    print(f"Wrote {ftgs_path} ({size_mb:.1f} MB)")

    # ─── Write scene_meta.json ───
    scene_center = np.median(means, axis=0).tolist()
    dists = np.linalg.norm(means - np.median(means, axis=0), axis=1)
    scene_radius = float(np.percentile(dists, 95))

    meta: dict = {
        "center": scene_center,
        "radius": scene_radius,
        "numSplats": N,
        "totalFrames": args.total_frames,
    }

    # ─── Optional: COLMAP cameras ───
    if args.colmap_dir:
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from datasets.read_write_model import read_model, qvec2rotmat
            from datasets.normalize import (
                similarity_from_cameras,
                transform_cameras,
                transform_points,
                align_principle_axes,
            )

            print(f"\nLoading COLMAP cameras from {args.colmap_dir} ...")
            cameras, images, points3D_dict = read_model(args.colmap_dir)
            bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
            sorted_images = sorted(images.values(), key=lambda im: im.name)

            camtoworlds = []
            camera_info = []
            Ks = []
            for im in sorted_images:
                R = qvec2rotmat(im.qvec)
                t = im.tvec.reshape(3, 1)
                w2c = np.concatenate([np.concatenate([R, t], 1), bottom], axis=0)
                c2w = np.linalg.inv(w2c)
                camtoworlds.append(c2w)
                cam = cameras[im.camera_id]
                params = cam.params
                w, h = cam.width, cam.height
                model_name = cam.model
                if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
                    fx = fy = params[0]
                elif model_name in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                                    "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
                    fx, fy = params[0], params[1]
                else:
                    fx = fy = params[0]
                K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
                Ks.append(K)
                camera_info.append({"name": im.name, "width": int(w), "height": int(h),
                                    "fx": float(fx), "fy": float(fy)})

            camtoworlds = np.array(camtoworlds)
            points3D = np.array([p.xyz for p in points3D_dict.values()], dtype=np.float32)

            # Normalize (same as training)
            T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            points3D = transform_points(T1, points3D)
            T2 = align_principle_axes(points3D)
            camtoworlds = transform_cameras(T2, camtoworlds)

            # lookatCenter
            directions, origins = camtoworlds[:, :3, 2:3], camtoworlds[:, :3, 3:4]
            m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
            mt_m = np.transpose(m, [0, 2, 1]) @ m
            focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
            meta["lookatCenter"] = focus_pt.tolist()

            # Camera presets (up to 12)
            n_cams = len(camtoworlds)
            max_presets = 12
            indices = (
                list(range(n_cams))
                if n_cams <= max_presets
                else np.linspace(0, n_cams - 1, max_presets, dtype=int).tolist()
            )
            presets = []
            for i in indices:
                c2w = camtoworlds[i]
                info = camera_info[i]
                fov_y = 2 * math.atan(info["height"] / (2 * Ks[i][0, 0])) * (180 / math.pi)
                presets.append({
                    "name": os.path.splitext(info["name"])[0],
                    "position": c2w[:3, 3].tolist(),
                    "worldMatrix": c2w.T.flatten().tolist(),
                    "fov": round(fov_y, 2),
                })
            meta["cameraPresets"] = presets
            print(f"  {len(presets)} camera presets exported")
        except Exception as e:
            print(f"  [WARNING] COLMAP load failed: {e}")

    meta_path = os.path.join(args.output_dir, "scene_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    print(f"\n使い方:")
    print(f"  cd simple-viewer && npm run dev")
    print(f"  ブラウザで http://localhost:5173/ を開く")


if __name__ == "__main__":
    main()
