"""
Convert FreeTimeGS 4D checkpoint (.pt) into assets for the web viewer.

Outputs:
- canonical.ply: canonical 3DGS splats (static parameters)
- canonical.sog: optional SOG v2 bundle (when --sog is specified)
- params_4d.bin: custom binary with velocities, times, durations
- scene_meta.json: scene metadata, camera presets, viewer params

Usage:
    # Basic (no camera presets):
    python scripts/convert_ckpt_to_web.py \
        --ckpt results/freetime_4d/ckpts/ckpt_29999.pt \
        --output-dir web-viewer/public/data

    # With COLMAP cameras → lookatCenter + camera presets (recommended):
    python scripts/convert_ckpt_to_web.py \
        --ckpt results/freetime_4d/ckpts/ckpt_29999.pt \
        --output-dir web-viewer/public/data \
        --colmap-dir dataset/dance/colmap/sparse/0

    # With SOG compression:
    python scripts/convert_ckpt_to_web.py \
        --ckpt results/freetime_4d/ckpts/ckpt_29999.pt \
        --output-dir web-viewer/public/data \
        --colmap-dir dataset/dance/colmap/sparse/0 \
        --sog
"""

import argparse
import json
import math
import os
import shlex
import shutil
import struct
import subprocess
import sys
from typing import List, Tuple

import numpy as np
import torch
from gsplat.exporter import export_splats

# Add project root to path for datasets module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def compute_spatial_mask(
    means: np.ndarray,
    use_spatial_filter: bool,
    spatial_filter_percentile: float,
    spatial_filter_padding: float,
) -> np.ndarray:
    if not use_spatial_filter:
        return np.ones((means.shape[0],), dtype=bool)

    min_coords = np.percentile(means, 5, axis=0)
    max_coords = np.percentile(means, 95, axis=0)
    scene_center = (min_coords + max_coords) / 2.0

    distances = np.linalg.norm(means - scene_center, axis=1)
    radius_percentile = np.percentile(distances, spatial_filter_percentile)
    filter_radius = radius_percentile * spatial_filter_padding

    return distances <= filter_radius


def compute_base_opacity_mask(opacities_logit: np.ndarray, threshold: float) -> np.ndarray:
    base_opacity = 1.0 / (1.0 + np.exp(-opacities_logit))
    return base_opacity >= threshold


def compute_max_scale_mask(scales_log: np.ndarray, max_scale: float) -> np.ndarray:
    """Remove splats whose largest exp(scale) exceeds *max_scale*."""
    if max_scale <= 0:
        return np.ones(scales_log.shape[0], dtype=bool)
    exp_scales = np.exp(np.clip(scales_log, -20, 20))
    return exp_scales.max(axis=1) <= max_scale


def compute_aspect_ratio_mask(scales_log: np.ndarray, max_ratio: float) -> np.ndarray:
    """Remove needle-like splats whose max/min scale ratio exceeds *max_ratio*."""
    if max_ratio <= 0:
        return np.ones(scales_log.shape[0], dtype=bool)
    exp_scales = np.exp(np.clip(scales_log, -20, 20))
    ratio = exp_scales.max(axis=1) / (exp_scales.min(axis=1) + 1e-10)
    return ratio <= max_ratio


def compute_temporal_visibility_mask(
    times: np.ndarray,
    durations_log: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Remove splats that are never visible during t=[0,1].

    For each splat the peak temporal opacity within [0, 1] is:
        max_opa = exp(-0.5 * ((clamp(mu_t, 0, 1) - mu_t) / s)^2)
    """
    mu_t = times.reshape(-1)
    s = np.exp(durations_log.reshape(-1)).astype(np.float64)
    s = np.clip(s, 0.02, None)

    t_nearest = np.clip(mu_t, 0.0, 1.0)
    dt = t_nearest - mu_t
    max_opa = np.exp(-0.5 * (dt ** 2) / (s ** 2 + 1e-8))
    return max_opa >= threshold


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ─── Camera utilities (from traj.py / FreeTime_dataset.py) ───────────────────

def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses (from traj.py).

    This gives a much better "look-at" target than simple centroid of Gaussians.
    """
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def load_colmap_cameras(colmap_dir: str) -> Tuple[np.ndarray, np.ndarray, List[dict], np.ndarray]:
    """Load camera poses and intrinsics from COLMAP sparse model.

    Returns:
        camtoworlds: [N, 4, 4] camera-to-world matrices (normalized to match training space)
        Ks: [N, 3, 3] intrinsic matrices (first camera shared)
        camera_info: list of dicts with name, width, height, fx, fy
        points3D: [M, 3] COLMAP 3D points (used internally for normalization)
    """
    from datasets.read_write_model import (
        read_model,
        qvec2rotmat,
    )
    from datasets.normalize import (
        similarity_from_cameras,
        transform_cameras,
        transform_points,
        align_principle_axes,
    )

    cameras, images, points3D_dict = read_model(colmap_dir)
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)

    # Sort by image name for deterministic ordering
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
        model_name = cam.model
        params = cam.params
        w, h = cam.width, cam.height

        # Extract focal length based on camera model
        if model_name in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        elif model_name in ("PINHOLE", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
                            "FULL_OPENCV", "THIN_PRISM_FISHEYE"):
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = w / 2, h / 2

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks.append(K)
        camera_info.append({
            "name": im.name,
            "width": int(w),
            "height": int(h),
            "fx": float(fx),
            "fy": float(fy),
        })

    camtoworlds = np.array(camtoworlds)

    # Load 3D points from COLMAP for normalization
    points3D = np.array([p.xyz for p in points3D_dict.values()], dtype=np.float32)
    print(f"  Loaded {len(points3D)} COLMAP 3D points for normalization")

    # Apply the same normalization as FreeTimeParser (datasets/FreeTime_dataset.py)
    # This ensures cameras match the normalized Gaussian space in the checkpoint
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    points3D = transform_points(T1, points3D)

    T2 = align_principle_axes(points3D)
    camtoworlds = transform_cameras(T2, camtoworlds)
    points3D = transform_points(T2, points3D)

    print(f"  Applied scene normalization (similarity + PCA alignment)")

    return camtoworlds, np.array(Ks), camera_info, points3D


def make_camera_presets(
    camtoworlds: np.ndarray,
    Ks: np.ndarray,
    camera_info: List[dict],
    max_presets: int = 12,
) -> List[dict]:
    """Create camera presets from COLMAP cameras for the web viewer.

    Selects evenly-spaced cameras and converts their c2w matrices to
    column-major float[16] for PlayCanvas Mat4.set().
    """
    n = len(camtoworlds)
    if n <= max_presets:
        indices = list(range(n))
    else:
        indices = np.linspace(0, n - 1, max_presets, dtype=int).tolist()

    presets = []
    for i in indices:
        c2w = camtoworlds[i]  # [4, 4] row-major
        info = camera_info[i]
        K = Ks[i]
        fx = K[0, 0]
        h = info["height"]
        fov_y = 2 * math.atan(h / (2 * fx)) * (180 / math.pi)

        # PlayCanvas Mat4.set() expects column-major order
        world_matrix = c2w.T.flatten().tolist()

        name = os.path.splitext(info["name"])[0]
        presets.append({
            "name": name,
            "position": c2w[:3, 3].tolist(),
            "worldMatrix": world_matrix,
            "fov": round(fov_y, 2),
        })
    return presets


def write_params_bin(
    output_path: str,
    velocities: np.ndarray,
    times: np.ndarray,
    durations_log: np.ndarray,
    total_frames: int,
    tex_width: int = 4096,
) -> Tuple[int, int]:
    num_splats = velocities.shape[0]
    tex_height = int(math.ceil(num_splats / tex_width))

    times_flat = times.reshape(-1).astype(np.float32)
    durations = np.exp(durations_log.reshape(-1)).astype(np.float32)
    durations = np.clip(durations, 0.02, None)
    velocities_f32 = velocities.astype(np.float32)

    header = struct.pack("<4I", num_splats, total_frames, tex_width, tex_height)

    with open(output_path, "wb") as f:
        f.write(header)
        f.write(velocities_f32.tobytes(order="C"))
        f.write(times_flat.tobytes(order="C"))
        f.write(durations.tobytes(order="C"))

    return tex_width, tex_height


def resolve_sog_command(custom_cli: str | None) -> List[str]:
    if custom_cli:
        return shlex.split(custom_cli)
    if shutil.which("splat-transform"):
        return ["splat-transform"]
    if shutil.which("npx"):
        return ["npx", "--yes", "@playcanvas/splat-transform"]
    raise RuntimeError(
        "splat-transform が見つかりません。npm で @playcanvas/splat-transform をインストールするか、"
        "--sog-cli で実行コマンドを指定してください。"
    )


def convert_ply_to_sog(
    ply_path: str,
    sog_path: str,
    iterations: int,
    gpu: str | None,
    cli: str | None,
) -> None:
    command = resolve_sog_command(cli)
    args = command + [
        "-i",
        str(iterations),
    ]
    if gpu is not None:
        args += ["-g", gpu]
    args += [ply_path, sog_path]

    result = subprocess.run(args, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "splat-transform で SOG 変換に失敗しました。\n"
            f"Command: {' '.join(args)}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FreeTimeGS 4D checkpoint to web assets")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pt")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--total-frames", type=int, default=60, help="Total frames in sequence (SelfCap=60, Neural3DV=300)")
    parser.add_argument("--spatial-percentile", type=float, default=95.0, help="Spatial filter percentile")
    parser.add_argument("--spatial-padding", type=float, default=1.1, help="Spatial filter padding")
    parser.add_argument("--no-spatial-filter", action="store_true", help="Disable spatial filtering")
    parser.add_argument("--base-opacity-threshold", type=float, default=0.005, help="Base opacity threshold")
    parser.add_argument("--temporal-threshold", type=float, default=0.01,
                        help="Temporal visibility threshold — splats never visible in [0,1] are removed")
    parser.add_argument("--sh-degree", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Max SH degree to export (0=DC only, greatly reduces PLY size)")
    parser.add_argument("--max-splats", type=int, default=0,
                        help="Random downsample to at most N splats (0=no limit)")
    parser.add_argument("--max-scale", type=float, default=0,
                        help="Max allowed exp(scale) per axis; splats exceeding this are removed "
                             "(0=no limit, matching trainer behavior — RECOMMENDED)")
    parser.add_argument("--max-aspect-ratio", type=float, default=0,
                        help="Max allowed scale aspect ratio (max/min); needle-like splats exceeding this are removed "
                             "(0=no limit, matching trainer behavior — RECOMMENDED)")
    parser.add_argument("--sog", action="store_true", help="Also export canonical.sog using splat-transform")
    parser.add_argument("--sog-iterations", type=int, default=10, help="SOG compression iterations")
    parser.add_argument("--sog-gpu", type=str, default=None, help="GPU adapter index or 'cpu' for SOG compression")
    parser.add_argument("--sog-cli", type=str, default=None, help="Custom splat-transform command")
    parser.add_argument("--colmap-dir", type=str, default=None,
                        help="Path to COLMAP sparse model (e.g. dataset/dance/colmap/sparse/0). "
                             "Enables lookatCenter and camera presets in scene_meta.json")

    args = parser.parse_args()

    ensure_dir(args.output_dir)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    means = splats["means"].cpu().numpy()
    scales_log = splats["scales"].cpu().numpy()
    quats = splats["quats"].cpu().numpy()
    opacities_logit = splats["opacities"].cpu().numpy()
    sh0 = splats["sh0"].cpu().numpy()
    shN = splats["shN"].cpu().numpy()
    velocities = splats["velocities"].cpu().numpy()
    times = splats["times"].cpu().numpy()
    durations_log = splats["durations"].cpu().numpy()

    # Normalize quaternions (gsplat PLY export does NOT normalize)
    quat_norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quat_norms = np.clip(quat_norms, 1e-8, None)
    quats = quats / quat_norms
    bad_quats = (np.abs(quat_norms.ravel() - 1.0) > 0.01).sum()
    print(f"Quaternion normalization: {bad_quats:,} / {quats.shape[0]:,} needed fixing")

    spatial_mask = compute_spatial_mask(
        means,
        use_spatial_filter=not args.no_spatial_filter,
        spatial_filter_percentile=args.spatial_percentile,
        spatial_filter_padding=args.spatial_padding,
    )
    opacity_mask = compute_base_opacity_mask(opacities_logit, args.base_opacity_threshold)
    temporal_mask = compute_temporal_visibility_mask(
        times, durations_log, threshold=args.temporal_threshold,
    )
    scale_mask = compute_max_scale_mask(scales_log, args.max_scale)
    aspect_mask = compute_aspect_ratio_mask(scales_log, args.max_aspect_ratio)
    mask = spatial_mask & opacity_mask & temporal_mask & scale_mask & aspect_mask
    print(f"Spatial: {spatial_mask.sum():,}, Opacity: {opacity_mask.sum():,}, "
          f"Temporal: {temporal_mask.sum():,}, Scale: {scale_mask.sum():,}, "
          f"Aspect: {aspect_mask.sum():,}, Combined: {mask.sum():,}")

    means = means[mask]
    scales_log = scales_log[mask]
    quats = quats[mask]
    opacities_logit = opacities_logit[mask]
    sh0 = sh0[mask]
    shN = shN[mask]
    velocities = velocities[mask]
    times = times[mask]
    durations_log = durations_log[mask]

    # --- Priority-based downsampling ---
    # Score each splat by importance = base_opacity × temporal_integral_over_[0,1]
    # temporal_integral = s√(2π) × [Φ((1-μ)/s) − Φ(−μ/s)]
    # Splats with high opacity AND broad temporal coverage are kept first.
    if args.max_splats > 0 and means.shape[0] > args.max_splats:
        from scipy.stats import norm as _norm

        _bo = 1.0 / (1.0 + np.exp(-opacities_logit.reshape(-1)))
        _mu = times.reshape(-1).astype(np.float64)
        _s = np.exp(durations_log.reshape(-1).astype(np.float64))
        _s = np.clip(_s, 0.02, None)

        _temporal_integral = _s * np.sqrt(2 * np.pi) * (
            _norm.cdf((1.0 - _mu) / _s) - _norm.cdf(-_mu / _s)
        )
        _importance = _bo * _temporal_integral

        idx = np.argsort(_importance)[::-1][:args.max_splats]
        idx.sort()
        means = means[idx]
        scales_log = scales_log[idx]
        quats = quats[idx]
        opacities_logit = opacities_logit[idx]
        sh0 = sh0[idx]
        shN = shN[idx]
        velocities = velocities[idx]
        times = times[idx]
        durations_log = durations_log[idx]

        # Report estimated visible splats per frame
        _mu2 = times.reshape(-1).astype(np.float64)
        _s2 = np.exp(durations_log.reshape(-1).astype(np.float64))
        _s2 = np.clip(_s2, 0.02, None)
        for _t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            _g = np.exp(-0.5 * ((_t - _mu2) / _s2) ** 2)
            print(f"  t={_t:.2f}: {(_g > 0.01).sum():,} visible splats (of {means.shape[0]:,})")

        print(f"Priority-downsampled to {means.shape[0]:,} splats (max_splats={args.max_splats:,})")

    # --- SH degree truncation ---
    sh_degree_coeffs = {0: 0, 1: 3, 2: 8, 3: 15}  # K = (L+1)^2 - 1
    max_k = sh_degree_coeffs[args.sh_degree]
    if shN.shape[1] > max_k:
        shN = shN[:, :max_k, :]
        print(f"SH truncated to degree {args.sh_degree} (K={max_k})")

    print(f"Selected {means.shape[0]:,} / {mask.shape[0]:,} Gaussians")

    # Diagnostic stats for verification
    base_opacity = 1.0 / (1.0 + np.exp(-opacities_logit))
    durations_exp = np.exp(durations_log.reshape(-1))
    print(f"  Velocity magnitude: [{np.linalg.norm(velocities, axis=1).min():.4f}, {np.linalg.norm(velocities, axis=1).max():.4f}]")
    print(f"  Times range: [{times.min():.4f}, {times.max():.4f}]")
    print(f"  Durations (exp) range: [{durations_exp.min():.4f}, {durations_exp.max():.4f}]")
    print(f"  Base opacity range: [{base_opacity.min():.4f}, {base_opacity.max():.4f}]")

    # NOTE: export_splats (PLY format) expects pre-activation values:
    #   scales  -> log-space  (viewer applies exp() on load)
    #   opacities -> logit    (viewer applies sigmoid() on load)
    # Do NOT apply exp/sigmoid here; that would cause double activation.

    ply_path = os.path.join(args.output_dir, "canonical.ply")
    export_splats(
        means=torch.from_numpy(means),
        scales=torch.from_numpy(scales_log.astype(np.float32)),
        quats=torch.from_numpy(quats),
        opacities=torch.from_numpy(opacities_logit.astype(np.float32)),
        sh0=torch.from_numpy(sh0.astype(np.float32)),
        shN=torch.from_numpy(shN.astype(np.float32)),
        format="ply",
        save_to=ply_path,
    )
    print(f"Wrote {ply_path}")

    # --- Scene metadata for camera auto-positioning ---
    scene_center = np.median(means, axis=0).tolist()
    dists = np.linalg.norm(means - np.median(means, axis=0), axis=1)
    scene_radius = float(np.percentile(dists, 95))

    # --- Compute recommended viewer parameters from actual data ---
    # Goal: match the Python trainer/viewer as closely as possible.
    # The trainer applies NO cap on scale or displacement.
    # We set generous soft caps that cover 99.9% of splats.
    exp_scales = np.exp(np.clip(scales_log, -20, 20))
    max_scale_p999 = float(np.percentile(exp_scales.max(axis=1), 99.9))
    # Use 1.5x 99.9th percentile — soft cap in shader, not removal
    recommended_max_scale = max(round(max_scale_p999 * 1.5, 3), 0.3)

    vel_mags = np.linalg.norm(velocities, axis=1)
    times_flat = times.reshape(-1)
    max_dt = np.maximum(np.abs(0 - times_flat), np.abs(1 - times_flat))
    max_disps = vel_mags * max_dt
    max_disp_p999 = float(np.percentile(max_disps, 99.9))
    # Use 2x 99.9th percentile — generous to avoid clipping motion
    recommended_max_disp = max(round(max_disp_p999 * 2.0, 2), 2.0)

    aspect_ratios = exp_scales.max(axis=1) / (exp_scales.min(axis=1) + 1e-10)
    recommended_max_aspect = float(min(np.percentile(aspect_ratios, 99.5), 200.0))

    print(f"  Recommended viewer params (trainer-matching):")
    print(f"    maxSplatScale: {recommended_max_scale} (p99.9={max_scale_p999:.4f})")
    print(f"    maxDisplacement: {recommended_max_disp} (p99.9={max_disp_p999:.4f})")
    print(f"    maxAspectRatio: {recommended_max_aspect:.1f}")

    meta: dict = {
        "center": scene_center,
        "radius": scene_radius,
        "numSplats": int(means.shape[0]),
        "totalFrames": args.total_frames,
        "viewerParams": {
            "maxSplatScale": recommended_max_scale,
            "maxDisplacement": recommended_max_disp,
            "maxAspectRatio": round(recommended_max_aspect, 1),
            "temporalThreshold": 0.01,
        },
    }

    # --- Load COLMAP cameras → lookatCenter + camera presets ---
    if args.colmap_dir:
        try:
            print(f"\nLoading COLMAP cameras from {args.colmap_dir} ...")
            camtoworlds, Ks, camera_info, _ = load_colmap_cameras(args.colmap_dir)
            print(f"  Loaded {len(camtoworlds)} cameras (normalized to training space)")

            # Compute lookatCenter (same as traj.py's focus_point_fn)
            poses_34 = camtoworlds[:, :3, :]  # [N, 3, 4]
            lookat_center = focus_point_fn(poses_34).tolist()
            meta["lookatCenter"] = lookat_center
            print(f"  lookatCenter (focus_point_fn): "
                  f"({lookat_center[0]:.3f}, {lookat_center[1]:.3f}, {lookat_center[2]:.3f})")

            # Camera presets
            presets = make_camera_presets(camtoworlds, Ks, camera_info, max_presets=12)
            meta["cameraPresets"] = presets
            print(f"  Exported {len(presets)} camera presets: "
                  f"{[p['name'] for p in presets]}")

        except Exception as e:
            print(f"  [WARNING] Failed to load COLMAP cameras: {e}")
            print(f"  Continuing without camera presets...")
    else:
        print("\n[NOTE] --colmap-dir not specified. No camera presets or lookatCenter.")
        print("  To get training-quality camera views, re-run with:")
        print(f"    --colmap-dir dataset/dance/colmap/sparse/0")

    meta_path = os.path.join(args.output_dir, "scene_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")

    if args.sog:
        sog_path = os.path.join(args.output_dir, "canonical.sog")
        convert_ply_to_sog(
            ply_path=ply_path,
            sog_path=sog_path,
            iterations=args.sog_iterations,
            gpu=args.sog_gpu,
            cli=args.sog_cli,
        )
        print(f"Wrote {sog_path}")

    bin_path = os.path.join(args.output_dir, "params_4d.bin")
    tex_width, tex_height = write_params_bin(
        bin_path,
        velocities=velocities,
        times=times,
        durations_log=durations_log,
        total_frames=args.total_frames,
        tex_width=4096,
    )
    print(f"Wrote {bin_path}")
    print(f"Params texture size: {tex_width} x {tex_height}")


if __name__ == "__main__":
    main()
