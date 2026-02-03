#!/usr/bin/env python3
"""
RoMa-based multi-view triangulation to per-frame NPY files.

This script matches a reference camera to all other cameras per frame
using RoMa, triangulates 3D points with COLMAP intrinsics/extrinsics,
then saves:
  - points3d_frameXXXXXX.npy
  - colors_frameXXXXXX.npy

Supports both flat image layout (images/0000_frame000000.png)
and folder layout (images/cam00/000000.png).
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from romatch import roma_outdoor

from datasets.read_write_model import read_cameras_binary, read_images_binary, qvec2rotmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triangulate RoMa matches into per-frame NPY point clouds")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset dir containing images/ and sparse/0")
    parser.add_argument("--out-dir", type=str, required=True, help="Output dir for NPY files")
    parser.add_argument("--frame-start", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--frame-end", type=int, default=60, help="End frame index (inclusive)")
    parser.add_argument("--ref-cam", type=str, default="0000", help="Reference camera id (e.g., 0000)")
    parser.add_argument("--certainty-th", type=float, default=0.3, help="RoMa certainty threshold")
    parser.add_argument("--use-ransac", action="store_true", help="Enable USAC_MAGSAC filtering")
    parser.add_argument("--ransac-th", type=float, default=0.5, help="RANSAC reprojection threshold (px)")
    parser.add_argument("--min-depth", type=float, default=1e-4, help="Minimum positive depth")
    parser.add_argument("--coarse-res", type=int, default=560, help="RoMa coarse resolution")
    parser.add_argument("--upsample-res", type=int, nargs=2, default=(864, 1152), help="RoMa upsample resolution (H W)")
    parser.add_argument("--device", type=str, default="cuda", help="cuda|cpu|mps")
    return parser.parse_args()


def build_camera_index(images_dir: Path) -> Tuple[Dict[str, Dict[int, Path]], List[str]]:
    """Return mapping cam_id -> {frame_idx: path}, and sorted cam_ids."""
    frame_re = re.compile(r"^(?P<cam>\d{4})_frame(?P<frame>\d{6})\.(png|jpg|jpeg)$")
    cam_map: Dict[str, Dict[int, Path]] = {}

    if any(p.is_dir() for p in images_dir.iterdir()):
        for cam_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
            cam_id = cam_dir.name.replace("cam", "") if cam_dir.name.startswith("cam") else cam_dir.name
            cam_map.setdefault(cam_id, {})
            for img_path in cam_dir.iterdir():
                if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                    continue
                stem = img_path.stem
                if stem.isdigit():
                    frame_idx = int(stem)
                    cam_map[cam_id][frame_idx] = img_path
    else:
        for img_path in images_dir.iterdir():
            if img_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                continue
            m = frame_re.match(img_path.name)
            if not m:
                continue
            cam_id = m.group("cam")
            frame_idx = int(m.group("frame"))
            cam_map.setdefault(cam_id, {})
            cam_map[cam_id][frame_idx] = img_path

    cam_ids = sorted(cam_map.keys())
    return cam_map, cam_ids


def cam_params_to_K(camera) -> np.ndarray:
    if camera.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        f, cx, cy = camera.params[:3]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif camera.model in ["PINHOLE"]:
        fx, fy, cx, cy = camera.params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif camera.model in ["OPENCV", "OPENCV_FISHEYE", "RADIAL", "RADIAL_FISHEYE"]:
        fx, fy, cx, cy = camera.params[:4]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        raise ValueError(f"Unsupported camera model: {camera.model}")
    return K


def undistort_points(kpts: np.ndarray, K: np.ndarray, params: np.ndarray, model: str) -> np.ndarray:
    if len(params) == 0:
        return kpts
    kpts = kpts.reshape(-1, 1, 2)
    if "FISHEYE" in model:
        undistorted = cv2.fisheye.undistortPoints(kpts, K, params[:4])
    else:
        undistorted = cv2.undistortPoints(kpts, K, params[:4])
    return undistorted.reshape(-1, 2)


def triangulate_pair(
    K1: np.ndarray,
    R1: np.ndarray,
    t1: np.ndarray,
    K2: np.ndarray,
    R2: np.ndarray,
    t2: np.ndarray,
    kpts1: np.ndarray,
    kpts2: np.ndarray,
    undistort: bool,
    params1: np.ndarray,
    params2: np.ndarray,
    model1: str,
    model2: str,
) -> np.ndarray:
    if undistort and (len(params1) > 0 or len(params2) > 0):
        k1 = undistort_points(kpts1, K1, params1, model1)
        k2 = undistort_points(kpts2, K2, params2, model2)
        P1 = np.hstack([R1, t1])
        P2 = np.hstack([R2, t2])
        pts4d = cv2.triangulatePoints(P1, P2, k1.T, k2.T)
    else:
        P1 = K1 @ np.hstack([R1, t1])
        P2 = K2 @ np.hstack([R2, t2])
        pts4d = cv2.triangulatePoints(P1, P2, kpts1.T, kpts2.T)
    pts3d = (pts4d[:3] / (pts4d[3:4] + 1e-8)).T
    return pts3d


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    images_dir = data_dir / "images"
    colmap_dir = data_dir / "sparse" / "0"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"images dir not found: {images_dir}")
    if not colmap_dir.exists():
        raise FileNotFoundError(f"COLMAP dir not found: {colmap_dir}")

    cam_map, cam_ids = build_camera_index(images_dir)
    if args.ref_cam not in cam_map:
        raise ValueError(f"Reference camera {args.ref_cam} not found in images")

    cameras = read_cameras_binary(str(colmap_dir / "cameras.bin"))
    images = read_images_binary(str(colmap_dir / "images.bin"))

    img_info = {}
    for img_id, img in images.items():
        name = img.name
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)
        img_info[name] = (img.camera_id, R, t)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    roma_model = roma_outdoor(device=device, coarse_res=args.coarse_res, upsample_res=tuple(args.upsample_res))
    model_h, model_w = roma_model.get_output_resolution()

    for frame_idx in range(args.frame_start, args.frame_end + 1):
        frame_tag = f"{frame_idx:06d}"

        ref_path = cam_map[args.ref_cam].get(frame_idx)
        if ref_path is None:
            print(f"[WARN] missing ref frame {frame_idx} ({args.ref_cam})")
            continue

        ref_img = Image.open(ref_path).convert("RGB")
        ref_w, ref_h = ref_img.size

        ref_name = ref_path.name
        if ref_name not in img_info:
            print(f"[WARN] COLMAP missing ref image: {ref_name}")
            continue

        ref_cam_id, R1, t1 = img_info[ref_name]
        cam1 = cameras[ref_cam_id]
        K1 = cam_params_to_K(cam1)
        params1 = cam1.params if hasattr(cam1, "params") else np.empty(0, dtype=np.float32)
        model1 = cam1.model

        points_list = []
        colors_list = []

        for cam_id in cam_ids:
            if cam_id == args.ref_cam:
                continue
            tgt_path = cam_map[cam_id].get(frame_idx)
            if tgt_path is None:
                continue

            tgt_name = tgt_path.name
            if tgt_name not in img_info:
                continue

            tgt_img = Image.open(tgt_path).convert("RGB")
            tgt_w, tgt_h = tgt_img.size

            cam_id2, R2, t2 = img_info[tgt_name]
            cam2 = cameras[cam_id2]
            K2 = cam_params_to_K(cam2)
            params2 = cam2.params if hasattr(cam2, "params") else np.empty(0, dtype=np.float32)
            model2 = cam2.model

            warp, certainty = roma_model.match(str(ref_path), str(tgt_path), device=device)
            matches, cert = roma_model.sample(warp, certainty)
            kptsA, kptsB = roma_model.to_pixel_coordinates(matches, model_h, model_w, model_h, model_w)

            kptsA = kptsA.cpu().numpy()
            kptsB = kptsB.cpu().numpy()
            cert = cert.cpu().numpy()

            good = cert > args.certainty_th
            kptsA = kptsA[good]
            kptsB = kptsB[good]

            if len(kptsA) < 8:
                continue

            # Scale coords to original resolution
            kptsA[:, 0] *= ref_w / model_w
            kptsA[:, 1] *= ref_h / model_h
            kptsB[:, 0] *= tgt_w / model_w
            kptsB[:, 1] *= tgt_h / model_h

            if args.use_ransac:
                F, mask = cv2.findFundamentalMat(
                    kptsA,
                    kptsB,
                    method=cv2.USAC_MAGSAC,
                    ransacReprojThreshold=args.ransac_th,
                    confidence=0.9999,
                    maxIters=10000,
                )
                if mask is not None:
                    mask = mask.squeeze().astype(bool)
                    kptsA = kptsA[mask]
                    kptsB = kptsB[mask]

            if len(kptsA) < 8:
                continue

            pts3d = triangulate_pair(
                K1,
                R1,
                t1,
                K2,
                R2,
                t2,
                kptsA,
                kptsB,
                undistort=True,
                params1=params1,
                params2=params2,
                model1=model1,
                model2=model2,
            )

            pts_cam1 = (R1 @ pts3d.T + t1).T
            pts_cam2 = (R2 @ pts3d.T + t2).T
            depth_ok = (pts_cam1[:, 2] > args.min_depth) & (pts_cam2[:, 2] > args.min_depth)
            pts3d = pts3d[depth_ok]
            kptsA = kptsA[depth_ok]

            if len(pts3d) == 0:
                continue

            ref_np = np.array(ref_img)
            kx = np.clip(np.round(kptsA[:, 0]).astype(int), 0, ref_w - 1)
            ky = np.clip(np.round(kptsA[:, 1]).astype(int), 0, ref_h - 1)
            colors = ref_np[ky, kx]

            points_list.append(pts3d.astype(np.float32))
            colors_list.append(colors.astype(np.float32))

        if not points_list:
            print(f"[WARN] no points at frame {frame_idx}")
            continue

        points = np.concatenate(points_list, axis=0)
        colors = np.concatenate(colors_list, axis=0)

        np.save(out_dir / f"points3d_frame{frame_idx:06d}.npy", points)
        np.save(out_dir / f"colors_frame{frame_idx:06d}.npy", colors)

        print(f"[OK] frame {frame_idx}: {len(points)} points")


if __name__ == "__main__":
    main()
