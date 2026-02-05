#!/usr/bin/env python3
"""
RoMa triangulation to per-frame NPY files.

For each frame, it:
- Loads a reference camera image and other camera images
- Matches using RoMa (romatch)
- Triangulates 3D points using fixed COLMAP camera poses
- Saves points3d_frameXXXXXX.npy and colors_frameXXXXXX.npy

This follows the paper's 4D initialization idea:
- Triangulate per-frame 3D points from multi-view images
- Initialize positions/time for that frame (velocity later)
"""

import argparse
import os
import re
import sys
import platform
from pathlib import Path
from typing import Dict, Tuple, Optional
from contextlib import nullcontext

import gc
import numpy as np
from PIL import Image

if platform.system() == "Darwin" and os.environ.get("KMP_DUPLICATE_LIB_OK") is None:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    print("[WARN] macOS OpenMP duplication detected. Set KMP_DUPLICATE_LIB_OK=TRUE as a workaround.")

try:
    import torch
except Exception as exc:
    raise RuntimeError("PyTorch is required to run RoMa triangulation.") from exc

from romatch import roma_outdoor, tiny_roma_v1_outdoor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.read_write_model import read_model, qvec2rotmat


SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoMa triangulation to NPY")
    parser.add_argument("--images-dir", help="Per-frame images dir")
    parser.add_argument("--colmap-model", help="COLMAP sparse model dir (e.g., sparse/0)")
    parser.add_argument("--output-dir", help="Output directory for NPY files")
    parser.add_argument("--data-dir", help="(legacy) dataset root dir containing images/ and sparse/0")
    parser.add_argument("--out-dir", help="(legacy) output dir for NPY files")
    parser.add_argument("--frame-start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frame-end", type=int, default=0, help="End frame index (inclusive). 0 means auto")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame step")
    parser.add_argument("--ref-cam", default="0000", help="Reference camera id (e.g., 0000)")
    parser.add_argument("--device", default="auto", help="Torch device (auto/cuda/mps/cpu)")
    parser.add_argument("--roma-model", choices=["roma_outdoor", "tiny_roma_v1_outdoor"], default="roma_outdoor")
    parser.add_argument("--certainty", type=float, default=0.3, help="RoMa certainty threshold")
    parser.add_argument(
        "--sample-mode",
        default="threshold",
        choices=["threshold", "threshold_balanced"],
        help="RoMa sample mode. 'threshold' avoids KDE OOM; 'threshold_balanced' may be higher quality but heavy.",
    )
    parser.add_argument("--sample-thresh", type=float, default=0.05, help="RoMa sample threshold")
    parser.add_argument("--use-ransac", action="store_true", help="Use RANSAC to filter matches (requires OpenCV)")
    parser.add_argument("--ransac-th", type=float, default=0.5, help="RANSAC reprojection threshold (px)")
    parser.add_argument("--max-matches", type=int, default=20000, help="Max matches per camera pair")
    parser.add_argument("--min-depth", type=float, default=1e-4, help="Min positive depth in both cameras")
    parser.add_argument("--max-reproj", type=float, default=2.0, help="Max reprojection error (px)")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Voxel size for optional downsampling (0 = off)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subsampling")
    parser.add_argument(
        "--image-scale",
        type=float,
        default=1.0,
        help="Scale input images for RoMa matching (e.g., 0.5 for half resolution).",
    )
    parser.add_argument("--amp", action="store_true", help="Enable AMP (fp16) during RoMa inference")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional cache directory for resized images (used when --image-scale != 1.0)",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir)
        if args.images_dir is None:
            args.images_dir = str(data_dir / "images")
        if args.colmap_model is None:
            args.colmap_model = str(data_dir / "sparse" / "0")
    if args.out_dir and args.output_dir is None:
        args.output_dir = args.out_dir

    missing = [name for name in ("images_dir", "colmap_model", "output_dir") if getattr(args, name) is None]
    if missing:
        raise ValueError(
            "Missing required arguments: "
            + ", ".join(missing)
            + ". Provide --images-dir --colmap-model --output-dir, or use --data-dir/--out-dir."
        )

    if args.frame_end == 0 and args.frame_start == 0:
        inferred = infer_max_frame(Path(args.images_dir), args.ref_cam)
        if inferred is not None:
            args.frame_end = inferred

    if args.frame_end < args.frame_start:
        inferred = infer_max_frame(Path(args.images_dir), args.ref_cam)
        if inferred is not None:
            args.frame_end = inferred

    if args.image_scale <= 0:
        raise ValueError("--image-scale must be > 0")

    args.device = resolve_device(args.device)

    return args


def infer_max_frame(images_dir: Path, ref_cam: str) -> Optional[int]:
    pattern = re.compile(rf"^{re.escape(ref_cam)}_frame(\d+)$")
    max_frame = None
    for path in images_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS:
            continue
        match = pattern.match(path.stem)
        if not match:
            continue
        frame_idx = int(match.group(1))
        if max_frame is None or frame_idx > max_frame:
            max_frame = frame_idx
    if max_frame is not None:
        print(f"[INFO] Auto-detected frame_end={max_frame} from images in {images_dir}")
    return max_frame


def resolve_device(requested: str) -> str:
    requested = requested.strip().lower().strip('"').strip("'")
    if requested == "":
        requested = "auto"

    normalized = requested.replace(" ", "")
    if "cuda" in normalized:
        normalized = "cuda"
    elif normalized in {"gpu", "nvidia"}:
        normalized = "cuda"

    cuda_ok = torch.cuda.is_available()
    mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    mps_has_current = hasattr(torch.mps, "current_device")

    print(
        "[INFO] Device check:",
        f"requested='{requested}'",
        f"normalized='{normalized}'",
        f"cuda_available={cuda_ok}",
        f"cuda_count={torch.cuda.device_count() if cuda_ok else 0}",
        f"mps_available={mps_ok}",
        f"mps_has_current={mps_has_current}",
    )

    if normalized == "auto":
        if cuda_ok:
            return "cuda"
        if mps_ok and mps_has_current:
            return "mps"
        return "cpu"

    if normalized == "cuda" and not cuda_ok:
        if mps_ok and mps_has_current:
            print("[WARN] CUDA not available. Falling back to MPS.")
            return "mps"
        print("[WARN] CUDA not available. Falling back to CPU.")
        return "cpu"

    if normalized == "cuda" and cuda_ok:
        return "cuda"

    if normalized == "mps":
        if not mps_ok or not mps_has_current:
            print("[WARN] MPS not available or incomplete. Falling back to CPU.")
            return "cpu"
        return "mps"

    if normalized == "cpu":
        return "cpu"

    print(f"[WARN] Unknown device '{requested}'. Falling back to CPU.")
    return "cpu"


def extract_cam_id(image_name: str) -> str:
    stem = Path(image_name).stem
    if stem.startswith("cam"):
        stem = stem[3:]
    digits = re.findall(r"\d+", stem)
    if digits:
        return digits[0].zfill(4)
    return stem


def camera_intrinsics(camera) -> np.ndarray:
    model = camera.model
    params = camera.params
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
        f, cx, cy = params[0], params[1], params[2]
        fx, fy = f, f
    elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "FOV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
    else:
        raise ValueError(f"Unsupported camera model: {model}")

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def build_camera_params(cameras, images, image_scale: float = 1.0) -> Dict[str, Dict[str, np.ndarray]]:
    params = {}
    for img in images.values():
        cam_id = extract_cam_id(img.name)
        cam = cameras[img.camera_id]
        K = camera_intrinsics(cam)
        if image_scale != 1.0:
            K[:2] *= image_scale
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)
        P = K @ np.hstack([R, t])
        params[cam_id] = {
            "K": K,
            "R": R,
            "t": t,
            "P": P,
            "width": int(round(cam.width * image_scale)),
            "height": int(round(cam.height * image_scale)),
        }
    return params


def find_frame_image(images_dir: Path, cam_id: str, frame_idx: int) -> Optional[Path]:
    for ext in SUPPORTED_IMAGE_EXTS:
        candidate = images_dir / f"{cam_id}_frame{frame_idx:06d}{ext}"
        if candidate.exists():
            return candidate
    return None


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def resized_image_path(path: Path, scale: float, cache_dir: Path) -> Path:
    scale_tag = f"s{scale:.3f}".replace(".", "p")
    return cache_dir / f"{path.stem}_{scale_tag}{path.suffix}"


def ensure_resized_image(path: Path, scale: float, cache_dir: Path) -> Path:
    if scale == 1.0:
        return path
    cache_dir.mkdir(parents=True, exist_ok=True)
    resized_path = resized_image_path(path, scale, cache_dir)
    if resized_path.exists():
        return resized_path
    with Image.open(path) as image:
        image = image.convert("RGB")
        w, h = image.size
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = image.resize((new_w, new_h), resample=Image.LANCZOS)
        resized.save(resized_path)
    return resized_path


def sample_colors(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    xs = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return image[ys, xs].astype(np.float32)


def triangulate_points(P1: np.ndarray, P2: np.ndarray, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    points = np.zeros((len(pts1), 3), dtype=np.float64)
    for i, (p1, p2) in enumerate(zip(pts1, pts2)):
        u1, v1 = p1
        u2, v2 = p2
        A = np.stack(
            [
                u1 * P1[2] - P1[0],
                v1 * P1[2] - P1[1],
                u2 * P2[2] - P2[0],
                v2 * P2[2] - P2[1],
            ],
            axis=0,
        )
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        X = X[:3] / X[3]
        points[i] = X
    return points


def reprojection_error(P: np.ndarray, points3d: np.ndarray, points2d: np.ndarray) -> np.ndarray:
    homog = np.hstack([points3d, np.ones((len(points3d), 1), dtype=np.float64)])
    proj = (P @ homog.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    return np.linalg.norm(proj - points2d, axis=1)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
    if voxel_size <= 0 or len(points) == 0:
        return points, colors
    coords = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(coords, axis=0, return_index=True)
    return points[unique_idx], colors[unique_idx]


def maybe_clear_device_cache(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    elif device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()


def amp_context(device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    if device == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if device == "mps":
        return torch.autocast(device_type="mps", dtype=torch.float16)
    return nullcontext()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cameras, images, _ = read_model(args.colmap_model)
    cam_params = build_camera_params(cameras, images, image_scale=args.image_scale)

    if args.ref_cam not in cam_params:
        raise ValueError(f"Reference camera {args.ref_cam} not found in COLMAP model")

    if args.roma_model == "roma_outdoor":
        roma_model = roma_outdoor(device=args.device)
    else:
        roma_model = tiny_roma_v1_outdoor(device=args.device)
    roma_model.sample_mode = args.sample_mode
    roma_model.sample_thresh = args.sample_thresh

    if args.use_ransac:
        try:
            import cv2  # type: ignore
        except Exception:
            print("[WARN] OpenCV not available; skipping RANSAC filtering.")
            cv2 = None
    else:
        cv2 = None

    frame_indices = list(range(args.frame_start, args.frame_end + 1, args.frame_step))

    if args.image_scale != 1.0:
        if args.cache_dir is None:
            scale_tag = f"s{args.image_scale:.3f}".replace(".", "p")
            cache_dir = output_dir / f"_roma_cache_{scale_tag}"
        else:
            cache_dir = Path(args.cache_dir)
        print(f"[INFO] Using image scale {args.image_scale} with cache dir: {cache_dir}")
    else:
        cache_dir = None

    for frame_idx in frame_indices:
        ref_path = find_frame_image(images_dir, args.ref_cam, frame_idx)
        if ref_path is None:
            print(f"[WARN] Missing reference image for frame {frame_idx:06d}. Skipping.")
            continue

        ref_image = None
        all_points = []
        all_colors = []

        try:
            ref_path_scaled = ensure_resized_image(ref_path, args.image_scale, cache_dir) if cache_dir else ref_path
            ref_image = load_image_rgb(ref_path_scaled)
            ref_h, ref_w = ref_image.shape[:2]

            for cam_id, params in cam_params.items():
                if cam_id == args.ref_cam:
                    continue

                other_path = find_frame_image(images_dir, cam_id, frame_idx)
                if other_path is None:
                    continue

                other_path_scaled = ensure_resized_image(other_path, args.image_scale, cache_dir) if cache_dir else other_path

                with torch.inference_mode(), amp_context(args.device, args.amp):
                    warp, certainty = roma_model.match(str(ref_path_scaled), str(other_path_scaled), device=args.device)
                    matches, cert = roma_model.sample(warp, certainty, num=args.max_matches)

                # Move to CPU early to avoid accumulating large GPU tensors across frames.
                matches = matches.detach().cpu()
                cert = cert.detach().cpu()
                del warp, certainty
                maybe_clear_device_cache(args.device)

                if matches is None or cert is None or len(matches) == 0:
                    continue

                cert = cert.squeeze()
                keep = cert >= args.certainty
                if keep.sum().item() == 0:
                    continue

                matches = matches[keep]
                cert = cert[keep]

                other_image = None
                try:
                    other_image = load_image_rgb(other_path_scaled)
                    other_h, other_w = other_image.shape[:2]

                    kpts_ref, kpts_other = roma_model.to_pixel_coordinates(matches, ref_h, ref_w, other_h, other_w)
                    kpts_ref = kpts_ref.numpy().astype(np.float64)
                    kpts_other = kpts_other.numpy().astype(np.float64)

                    if len(kpts_ref) > args.max_matches:
                        idx = np.random.choice(len(kpts_ref), args.max_matches, replace=False)
                        kpts_ref = kpts_ref[idx]
                        kpts_other = kpts_other[idx]

                    if cv2 is not None and args.use_ransac:
                        F, mask = cv2.findFundamentalMat(
                            kpts_ref,
                            kpts_other,
                            ransacReprojThreshold=args.ransac_th,
                            method=cv2.USAC_MAGSAC,
                            confidence=0.999,
                            maxIters=10000,
                        )
                        if mask is not None:
                            mask = mask.ravel().astype(bool)
                            kpts_ref = kpts_ref[mask]
                            kpts_other = kpts_other[mask]

                    if len(kpts_ref) == 0:
                        continue

                    P1 = cam_params[args.ref_cam]["P"]
                    P2 = params["P"]
                    R1, t1 = cam_params[args.ref_cam]["R"], cam_params[args.ref_cam]["t"]
                    R2, t2 = params["R"], params["t"]

                    points3d = triangulate_points(P1, P2, kpts_ref, kpts_other)

                    z1 = (R1 @ points3d.T + t1).T[:, 2]
                    z2 = (R2 @ points3d.T + t2).T[:, 2]
                    depth_mask = (z1 > args.min_depth) & (z2 > args.min_depth)

                    if depth_mask.sum() == 0:
                        continue

                    points3d = points3d[depth_mask]
                    kpts_ref = kpts_ref[depth_mask]
                    kpts_other = kpts_other[depth_mask]

                    err1 = reprojection_error(P1, points3d, kpts_ref)
                    err2 = reprojection_error(P2, points3d, kpts_other)
                    reproj_mask = (err1 < args.max_reproj) & (err2 < args.max_reproj)

                    if reproj_mask.sum() == 0:
                        continue

                    points3d = points3d[reproj_mask]
                    kpts_ref = kpts_ref[reproj_mask]

                    colors = sample_colors(ref_image, kpts_ref)

                    all_points.append(points3d.astype(np.float32))
                    all_colors.append(colors.astype(np.float32))
                    del kpts_ref, kpts_other, points3d, colors
                finally:
                    del other_image, matches, cert
                    maybe_clear_device_cache(args.device)

            if len(all_points) == 0:
                print(f"[WARN] No points generated for frame {frame_idx:06d}")
                continue

            points = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0)

            points, colors = voxel_downsample(points, colors, args.voxel_size)

            points_path = output_dir / f"points3d_frame{frame_idx:06d}.npy"
            colors_path = output_dir / f"colors_frame{frame_idx:06d}.npy"

            np.save(points_path, points)
            np.save(colors_path, colors)

            print(f"[Frame {frame_idx:06d}] points={len(points)} saved to {output_dir}")
        finally:
            del ref_image, all_points, all_colors
            maybe_clear_device_cache(args.device)


if __name__ == "__main__":
    main()
