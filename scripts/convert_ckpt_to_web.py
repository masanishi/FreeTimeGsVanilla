"""
Convert FreeTimeGS 4D checkpoint (.pt) into assets for the web viewer.

Outputs:
- canonical.ply: canonical 3DGS splats (static parameters)
- canonical.sog: optional SOG v2 bundle (when --sog is specified)
- params_4d.bin: custom binary with velocities, times, durations

Usage:
    python scripts/convert_ckpt_to_web.py \
        --ckpt results/freetime_4d/ckpts/ckpt_29999.pt \
        --output-dir web-viewer/public/data \
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
from typing import List, Tuple

import numpy as np
import torch
from gsplat.exporter import export_splats


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
    parser.add_argument("--max-scale", type=float, default=0.1,
                        help="Max allowed exp(scale) per axis; splats exceeding this are removed (0=no limit)")
    parser.add_argument("--max-aspect-ratio", type=float, default=50.0,
                        help="Max allowed scale aspect ratio (max/min); needle-like splats exceeding this are removed (0=no limit)")
    parser.add_argument("--sog", action="store_true", help="Also export canonical.sog using splat-transform")
    parser.add_argument("--sog-iterations", type=int, default=10, help="SOG compression iterations")
    parser.add_argument("--sog-gpu", type=str, default=None, help="GPU adapter index or 'cpu' for SOG compression")
    parser.add_argument("--sog-cli", type=str, default=None, help="Custom splat-transform command")

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
    # These match the trainer's behavior (no artificial caps)
    exp_scales = np.exp(np.clip(scales_log, -20, 20))
    max_scale_p99 = float(np.percentile(exp_scales.max(axis=1), 99.5))
    # Use 2x the 99.5th percentile as safe max, but at least 0.1
    recommended_max_scale = max(round(max_scale_p99 * 2, 3), 0.1)

    vel_mags = np.linalg.norm(velocities, axis=1)
    times_flat = times.reshape(-1)
    max_dt = np.maximum(np.abs(0 - times_flat), np.abs(1 - times_flat))
    max_disps = vel_mags * max_dt
    max_disp_p99 = float(np.percentile(max_disps, 99))
    # Use 1.5x the 99th percentile, but at least 1.0
    recommended_max_disp = max(round(max_disp_p99 * 1.5, 2), 1.0)

    aspect_ratios = exp_scales.max(axis=1) / (exp_scales.min(axis=1) + 1e-10)
    recommended_max_aspect = float(min(np.percentile(aspect_ratios, 99), 100.0))

    print(f"  Recommended viewer params:")
    print(f"    maxSplatScale: {recommended_max_scale}")
    print(f"    maxDisplacement: {recommended_max_disp}")
    print(f"    maxAspectRatio: {recommended_max_aspect:.1f}")

    meta = {
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
