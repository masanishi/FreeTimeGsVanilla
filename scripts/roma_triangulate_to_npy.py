#!/usr/bin/env python3
"""
RoMa triangulation to per-frame NPY for FreeTimeGS.

This script:
- Loads a reference COLMAP model (sparse/0) to get camera intrinsics/extrinsics
- For each frame: matches reference camera to other cameras
- Triangulates matches to 3D points
- Saves points3d_frame%06d.npy and colors_frame%06d.npy

Default matcher is ROMA (paper). ROMA is required.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from tqdm import tqdm

try:
    from pycolmap import Reconstruction as PyColmapReconstruction
    PYCOLMAP_API = "new"
except ImportError:
    try:
        from pycolmap import SceneManager
        PyColmapReconstruction = None
        PYCOLMAP_API = "old"
    except ImportError:
        SceneManager = None
        PyColmapReconstruction = None
        PYCOLMAP_API = None


def load_colmap_cameras(colmap_dir: str):
    if PYCOLMAP_API is None:
        raise RuntimeError("pycolmap not found. Install with: pip install pycolmap")

    if PYCOLMAP_API == "new":
        rec = PyColmapReconstruction()
        rec.read(colmap_dir)
        imdata = rec.images
        cameras_dict = rec.cameras
    else:
        manager = SceneManager(colmap_dir)
        manager.load_cameras()
        manager.load_images()
        imdata = manager.images
        cameras_dict = manager.cameras

    camera_map = {}
    for k in imdata:
        im = imdata[k]

        if PYCOLMAP_API == "new":
            pose = im.cam_from_world()
            quat = pose.rotation.quat  # [x, y, z, w]
            x, y, z, w = quat
            rot = np.array([
                [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
            ])
            trans = np.array(pose.translation).reshape(3, 1)
        else:
            rot = im.R()
            trans = im.tvec.reshape(3, 1)

        w2c = np.concatenate([np.concatenate([rot, trans], 1), np.array([[0, 0, 0, 1]])], axis=0)

        cam = cameras_dict[im.camera_id]
        if PYCOLMAP_API == "new":
            fx = cam.focal_length_x
            fy = cam.focal_length_y
            cx = cam.principal_point_x
            cy = cam.principal_point_y
            model_name = str(cam.model)
            cam_params = cam.params
        else:
            fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
            model_name = str(cam.camera_type)
            cam_params = []

        if "PINHOLE" in model_name or model_name in ["0", "1"]:
            params = np.zeros(0, dtype=np.float32)
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
            params = np.zeros(0, dtype=np.float32)

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        name = Path(im.name)
        cam_name = name.parent.name if ("/" in im.name or "\\" in im.name) else name.stem

        camera_map[cam_name] = {
            "K": K,
            "w2c": w2c,
            "dist": params,
        }

    return camera_map


def try_init_roma(device, amp):
    try:
        import romatch
    except Exception:
        return None

    print(f"[ROMA][DEBUG] romatch module path: {getattr(romatch, '__file__', 'N/A')}")
    print(f"[ROMA][DEBUG] available attrs: {', '.join([k for k in ['roma_outdoor', 'roma_indoor', 'tiny_roma_v1_outdoor'] if hasattr(romatch, k)])}")

    amp_dtype = torch.float16 if amp else torch.float32
    if hasattr(romatch, "roma_outdoor"):
        matcher = romatch.roma_outdoor(device=device, amp_dtype=amp_dtype, symmetric=False)
    elif hasattr(romatch, "roma_indoor"):
        matcher = romatch.roma_indoor(device=device, amp_dtype=amp_dtype, symmetric=False)
    elif hasattr(romatch, "tiny_roma_v1_outdoor"):
        matcher = romatch.tiny_roma_v1_outdoor(device=device)
    else:
        return None

    if hasattr(matcher, "eval"):
        matcher.eval()
    return matcher


def match_roma(matcher, img0, img1, cert_th=0.3, max_matches=20000):
    with torch.inference_mode():
        warp_t, certainty_t = matcher.match(img0, img1, batched=True)

    if warp_t.ndim == 4:
        warp_t = warp_t[0]
    if certainty_t.ndim == 3:
        certainty_t = certainty_t[0]

    warp = warp_t.detach().cpu().numpy()
    certainty = certainty_t.detach().cpu().numpy()
    del warp_t, certainty_t

    h, w, _ = warp.shape
    flat_warp = warp.reshape(-1, 4)
    flat_cert = certainty.reshape(-1)

    keep = flat_cert >= cert_th
    if keep.sum() == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)

    pts = flat_warp[keep]
    scores = flat_cert[keep]

    if len(scores) > max_matches:
        idx = np.random.choice(len(scores), max_matches, replace=False)
        pts = pts[idx]
        scores = scores[idx]

    ax = (pts[:, 0] + 1.0) * 0.5 * (w - 1)
    ay = (pts[:, 1] + 1.0) * 0.5 * (h - 1)
    bx = (pts[:, 2] + 1.0) * 0.5 * (w - 1)
    by = (pts[:, 3] + 1.0) * 0.5 * (h - 1)

    pts0 = np.stack([ax, ay], axis=1).astype(np.float32)
    pts1 = np.stack([bx, by], axis=1).astype(np.float32)
    scores = scores.astype(np.float32)
    return pts0, pts1, scores


def triangulate_pair(K0, w2c0, K1, w2c1, pts0, pts1):
    P0 = K0 @ w2c0[:3, :]
    P1 = K1 @ w2c1[:3, :]
    pts0_h = pts0.T
    pts1_h = pts1.T
    X_h = cv2.triangulatePoints(P0, P1, pts0_h, pts1_h)
    X = (X_h[:3] / X_h[3:4]).T
    return X


def sample_colors(img, pts):
    h, w = img.shape[:2]
    xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
    return img[ys, xs, :].astype(np.float32)


def format_memory_stats(device: str) -> str:
    if device == "cuda":
        try:
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return f"CUDA allocated={allocated / (1024 ** 2):.1f}MB, reserved={reserved / (1024 ** 2):.1f}MB"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--colmap-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=59)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--ref-cam", default="0000")
    parser.add_argument("--image-ext", default="png")
    parser.add_argument("--matcher", choices=["roma"], default="roma")
    parser.add_argument("--device", choices=["auto", "cuda", "mps"], default="auto")
    parser.add_argument("--certainty", type=float, default=0.3)
    parser.add_argument("--use-ransac", action="store_true")
    parser.add_argument("--ransac-th", type=float, default=0.5)
    parser.add_argument("--min-depth", type=float, default=1e-4)
    parser.add_argument("--max-matches", type=int, default=20000)
    parser.add_argument("--voxel-size", type=float, default=0.0)
    parser.add_argument("--image-scale", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-dir", default="")

    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    camera_map = load_colmap_cameras(args.colmap_model)
    camera_names = sorted(camera_map.keys())

    if args.ref_cam not in camera_map:
        raise ValueError(f"ref cam not found in COLMAP: {args.ref_cam}")

    if args.device == "auto":
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

    roma_matcher = try_init_roma(args.device, args.amp)
    if roma_matcher is None:
        raise RuntimeError("ROMA is not available. Install romatch and ensure GPU/MPS support is enabled.")
    print(f"[ROMA] matcher initialized on {args.device}")

    for frame_idx in tqdm(range(args.frame_start, args.frame_end + 1, args.frame_step), desc="Triangulating frames"):
        ref_path = images_dir / args.ref_cam / f"{frame_idx:06d}.{args.image_ext}"
        if not ref_path.exists():
            print(f"[WARN] missing ref frame: {ref_path}")
            continue

        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            print(f"[WARN] failed to read: {ref_path}")
            continue

        if args.image_scale != 1.0:
            ref_img = cv2.resize(ref_img, dsize=None, fx=args.image_scale, fy=args.image_scale)

        all_points = []
        all_colors = []

        for cam_name in camera_names:
            if cam_name == args.ref_cam:
                continue

            other_path = images_dir / cam_name / f"{frame_idx:06d}.{args.image_ext}"
            if not other_path.exists():
                continue

            other_img = cv2.imread(str(other_path))
            if other_img is None:
                continue

            if args.image_scale != 1.0:
                other_img = cv2.resize(other_img, dsize=None, fx=args.image_scale, fy=args.image_scale)

            ref_pil = Image.fromarray(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
            other_pil = Image.fromarray(cv2.cvtColor(other_img, cv2.COLOR_BGR2RGB))
            pts0, pts1, scores = match_roma(
                roma_matcher,
                ref_pil,
                other_pil,
                cert_th=args.certainty,
                max_matches=args.max_matches,
            )
            if len(scores) > 0:
                keep = scores >= args.certainty
                pts0, pts1, scores = pts0[keep], pts1[keep], scores[keep]

            if len(pts0) < 8:
                continue

            if args.use_ransac:
                F, mask = cv2.findFundamentalMat(pts0, pts1, cv2.FM_RANSAC, args.ransac_th, 0.999)
                if mask is not None:
                    mask = mask.squeeze().astype(bool)
                    pts0 = pts0[mask]
                    pts1 = pts1[mask]

            if len(pts0) < 8:
                continue

            # undo image scale for triangulation
            if args.image_scale != 1.0:
                pts0 = pts0 / args.image_scale
                pts1 = pts1 / args.image_scale

            K0 = camera_map[args.ref_cam]["K"]
            w2c0 = camera_map[args.ref_cam]["w2c"]
            K1 = camera_map[cam_name]["K"]
            w2c1 = camera_map[cam_name]["w2c"]

            X = triangulate_pair(K0, w2c0, K1, w2c1, pts0, pts1)

            # depth filter
            X_h = np.concatenate([X, np.ones((len(X), 1))], axis=1)
            depth0 = (w2c0 @ X_h.T).T[:, 2]
            depth1 = (w2c1 @ X_h.T).T[:, 2]
            valid = (depth0 > args.min_depth) & (depth1 > args.min_depth)
            X = X[valid]
            pts0 = pts0[valid]

            if len(X) == 0:
                continue

            colors = sample_colors(ref_img, pts0)
            all_points.append(X)
            all_colors.append(colors)

            del other_img, other_pil, ref_pil, pts0, pts1, scores, X, colors


        if len(all_points) == 0:
            print(f"[WARN] no points for frame {frame_idx}")
            continue

        points = np.concatenate(all_points, axis=0).astype(np.float32)
        colors = np.concatenate(all_colors, axis=0).astype(np.float32)

        if args.voxel_size and args.voxel_size > 0:
            vox = np.floor(points / args.voxel_size).astype(np.int64)
            key = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791
            _, unique_idx = np.unique(key, return_index=True)
            points = points[unique_idx]
            colors = colors[unique_idx]

        np.save(output_dir / f"points3d_frame{frame_idx:06d}.npy", points)
        np.save(output_dir / f"colors_frame{frame_idx:06d}.npy", colors)

        print(f"[Frame {frame_idx:06d}] points={len(points)} | {format_memory_stats(args.device)}")

        del points, colors, all_points, all_colors, ref_img


if __name__ == "__main__":
    main()
