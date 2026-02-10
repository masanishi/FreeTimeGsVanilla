#!/usr/bin/env python3
"""
Export checkpoint (.pt) to viewer-compatible files.

Produces exactly the binary formats expected by simple-viewer:
  1. canonical.ply   — Binary PLY (gsplat export_splats format)
  2. params_4d.bin   — Velocities + times + durations (binary)
  3. scene_meta.json — Scene metadata + optional camera presets

This script applies the SAME activation conventions as the official
simple_trainer and viewer_4d.py, avoiding the data conversion bugs
present in earlier ad-hoc scripts.

Activation conventions stored in PLY (log/logit space):
  - scales:    log-space  (viewer applies exp)
  - opacities: logit-space (viewer applies sigmoid)
  - quats:     raw (viewer normalizes)
  - SH:        channel-grouped via permute(0,2,1).reshape(N,-1)

params_4d.bin stores:
  - velocities: as-is [N*3]
  - times:      as-is [N] (squeezed from [N,1])
  - durations:  ACTIVATED  exp(log_dur), clamped min=0.02

Usage:
    python scripts/export_for_viewer.py \\
        --ckpt results/dance/ckpts/ckpt_49999.pt \\
        --output simple-viewer/public/data \\
        --total-frames 24

    # With camera presets from COLMAP:
    python scripts/export_for_viewer.py \\
        --ckpt ckpt_49999.pt \\
        --output simple-viewer/public/data \\
        --total-frames 24 \\
        --colmap-dir dataset/dance/colmap/sparse/0
"""

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

# Add project root for dataset imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def write_canonical_ply(path: str, splats: dict):
    """
    Write canonical Gaussian parameters to binary PLY.

    Format matches gsplat's export_splats() output:
      x, y, z, scale_0-2 (log), rot_0-3 (wxyz), opacity (logit),
      f_dc_0-2, f_rest_0..44 (channel-grouped)

    The viewer (plyParser.ts) will apply:
      - exp(scale) for scales
      - sigmoid(opacity) for opacities
      - normalize(quat) for quaternions
    """
    means = splats["means"].cpu().numpy().astype(np.float32)        # [N, 3]
    scales = splats["scales"].cpu().numpy().astype(np.float32)      # [N, 3] log-space
    quats = splats["quats"].cpu().numpy().astype(np.float32)        # [N, 4] wxyz
    opacities = splats["opacities"].cpu().numpy().astype(np.float32)  # [N] logit
    sh0 = splats["sh0"].cpu().numpy().astype(np.float32)            # [N, 1, 3]
    shN = splats["shN"].cpu().numpy().astype(np.float32)            # [N, K, 3]

    N = means.shape[0]
    K = shN.shape[1]  # SH coefficients per channel (15 for degree 3)

    # SH layout conversion:
    # Checkpoint: [N, K, 3] interleaved (coeff_k, channel_c)
    # PLY:        [N, 45] channel-grouped = permute(0,2,1).reshape(N,-1)
    #             → [R0..R14, G0..G14, B0..B14]
    shN_grouped = shN.transpose(0, 2, 1).reshape(N, -1)  # [N, K*3]
    # Pad to 45 if K < 15
    if shN_grouped.shape[1] < 45:
        pad = np.zeros((N, 45 - shN_grouped.shape[1]), dtype=np.float32)
        shN_grouped = np.concatenate([shN_grouped, pad], axis=1)

    sh0_flat = sh0.reshape(N, 3)  # [N, 3]

    # Build PLY header
    properties = ["x", "y", "z"]
    properties += ["scale_0", "scale_1", "scale_2"]
    properties += ["rot_0", "rot_1", "rot_2", "rot_3"]
    properties += ["opacity"]
    properties += ["f_dc_0", "f_dc_1", "f_dc_2"]
    properties += [f"f_rest_{i}" for i in range(45)]

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    for p in properties:
        header += f"property float {p}\n"
    header += "end_header\n"

    # Write binary data
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))

        for i in range(N):
            # Position (xyz)
            f.write(struct.pack("<fff", means[i, 0], means[i, 1], means[i, 2]))
            # Scales (log-space, viewer applies exp)
            f.write(struct.pack("<fff", scales[i, 0], scales[i, 1], scales[i, 2]))
            # Quaternion (w,x,y,z — viewer normalizes)
            f.write(struct.pack("<ffff", quats[i, 0], quats[i, 1], quats[i, 2], quats[i, 3]))
            # Opacity (logit-space, viewer applies sigmoid)
            f.write(struct.pack("<f", opacities[i]))
            # SH DC (as-is)
            f.write(struct.pack("<fff", sh0_flat[i, 0], sh0_flat[i, 1], sh0_flat[i, 2]))
            # SH higher-order (channel-grouped)
            f.write(struct.pack(f"<{45}f", *shN_grouped[i]))

    bytes_per_vertex = len(properties) * 4
    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  canonical.ply: {N:,} splats, {bytes_per_vertex} bytes/vertex, {file_size_mb:.1f} MB")


def write_canonical_ply_fast(path: str, splats: dict):
    """
    Fast vectorized version of write_canonical_ply using numpy bulk write.
    """
    means = splats["means"].cpu().numpy().astype(np.float32)
    scales = splats["scales"].cpu().numpy().astype(np.float32)
    quats = splats["quats"].cpu().numpy().astype(np.float32)
    opacities = splats["opacities"].cpu().numpy().astype(np.float32)
    sh0 = splats["sh0"].cpu().numpy().astype(np.float32)
    shN = splats["shN"].cpu().numpy().astype(np.float32)

    N = means.shape[0]
    K = shN.shape[1]

    # SH layout: [N,K,3] → [N,3,K] → [N,3*K] (channel-grouped)
    shN_grouped = shN.transpose(0, 2, 1).reshape(N, -1)
    if shN_grouped.shape[1] < 45:
        pad = np.zeros((N, 45 - shN_grouped.shape[1]), dtype=np.float32)
        shN_grouped = np.concatenate([shN_grouped, pad], axis=1)

    sh0_flat = sh0.reshape(N, 3)

    # Ensure opacities is 2D for concatenation
    if opacities.ndim == 1:
        opacities = opacities[:, np.newaxis]

    # Build per-vertex array: [N, 62] (3+3+4+1+3+45 = 59... let me count)
    # x,y,z (3) + scale0-2 (3) + rot0-3 (4) + opacity (1) + f_dc0-2 (3) + f_rest0-44 (45) = 59
    vertex_data = np.concatenate([
        means,           # 3
        scales,          # 3
        quats,           # 4
        opacities,       # 1
        sh0_flat,        # 3
        shN_grouped,     # 45
    ], axis=1)  # [N, 59]

    assert vertex_data.shape == (N, 59), f"Expected (N, 59), got {vertex_data.shape}"

    # Build header
    properties = ["x", "y", "z"]
    properties += ["scale_0", "scale_1", "scale_2"]
    properties += ["rot_0", "rot_1", "rot_2", "rot_3"]
    properties += ["opacity"]
    properties += ["f_dc_0", "f_dc_1", "f_dc_2"]
    properties += [f"f_rest_{i}" for i in range(45)]

    header = "ply\n"
    header += "format binary_little_endian 1.0\n"
    header += f"element vertex {N}\n"
    for p in properties:
        header += f"property float {p}\n"
    header += "end_header\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        vertex_data.tofile(f)

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  canonical.ply: {N:,} splats, {59 * 4} bytes/vertex, {file_size_mb:.1f} MB")


def write_params_4d_bin(path: str, splats: dict, total_frames: int):
    """
    Write 4D temporal parameters in binary format.

    Layout (little-endian):
      Header (16 bytes):
        uint32 numSplats
        uint32 totalFrames
        uint32 texWidth     (2048 — matches renderer TEX_WIDTH)
        uint32 texHeight    (ceil(N / 2048))
      Data:
        float32[N*3] velocities
        float32[N]   times      (mu_t, squeezed from [N,1])
        float32[N]   durations  (ACTIVATED: exp + clamp min=0.02)
    """
    velocities = splats["velocities"].cpu().numpy().astype(np.float32)  # [N, 3]
    times = splats["times"].cpu().numpy().astype(np.float32)            # [N, 1]
    durations_log = splats["durations"].cpu().numpy().astype(np.float32)  # [N, 1]

    N = velocities.shape[0]
    TEX_WIDTH = 2048
    tex_h = (N + TEX_WIDTH - 1) // TEX_WIDTH

    # Squeeze times/durations from [N,1] to [N]
    times_flat = times.squeeze(-1)
    # Activate durations: exp(log_dur), clamp min=0.02 (matches viewer_4d.py)
    durations_activated = np.maximum(np.exp(durations_log.squeeze(-1)), 0.02)

    with open(path, "wb") as f:
        # Header (16 bytes)
        f.write(struct.pack("<IIII", N, total_frames, TEX_WIDTH, tex_h))
        # Data
        velocities.ravel().tofile(f)
        times_flat.tofile(f)
        durations_activated.tofile(f)

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  params_4d.bin: {N:,} splats, {total_frames} frames, {file_size_mb:.1f} MB")


def write_ftgs(path: str, splats: dict, total_frames: int):
    """
    Write .ftgs binary (all-in-one format for simple-viewer's ptLoader.ts).

    Layout (little-endian):
      Header (32 bytes):
        uint32 magic = 0x46544753 ("FTGS")
        uint32 version = 1
        uint32 numSplats
        uint32 totalFrames
        uint32 shDegree
        uint32 shCoeffsPerChan (K)
        uint32[2] reserved

      Data (contiguous float32):
        means [N*3], scales [N*3] (log), quats [N*4] (wxyz),
        opacities [N] (logit), sh0 [N*3], shN [N*K*3] (interleaved),
        velocities [N*3], times [N], durations [N] (log)
    """
    means = splats["means"].cpu().numpy().astype(np.float32)
    scales = splats["scales"].cpu().numpy().astype(np.float32)
    quats = splats["quats"].cpu().numpy().astype(np.float32)
    opacities = splats["opacities"].cpu().numpy().astype(np.float32)
    sh0 = splats["sh0"].cpu().numpy().astype(np.float32)
    shN = splats["shN"].cpu().numpy().astype(np.float32)
    velocities = splats["velocities"].cpu().numpy().astype(np.float32)
    times = splats["times"].cpu().numpy().astype(np.float32)
    durations = splats["durations"].cpu().numpy().astype(np.float32)

    N = means.shape[0]
    K = shN.shape[1]
    sh_degree = {0: 0, 3: 1, 8: 2, 15: 3}.get(K, 3)

    with open(path, "wb") as f:
        # Header (32 bytes)
        f.write(struct.pack("<IIIIIIII",
            0x46544753,  # magic
            1,           # version
            N,
            total_frames,
            sh_degree,
            K,
            0, 0,        # reserved
        ))

        # Data sections (raw, no activation — ptLoader.ts does it)
        means.ravel().tofile(f)
        scales.ravel().tofile(f)
        quats.ravel().tofile(f)
        opacities.ravel().tofile(f)
        sh0.reshape(N, 3).tofile(f)    # [N, 1, 3] → [N, 3]
        shN.ravel().tofile(f)           # [N, K, 3] interleaved (ptLoader converts)
        velocities.ravel().tofile(f)
        times.squeeze(-1).tofile(f)     # [N, 1] → [N]
        durations.squeeze(-1).tofile(f) # [N, 1] → [N] (log-space, ptLoader activates)

    file_size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  checkpoint.ftgs: {N:,} splats, SH degree {sh_degree} (K={K}), {file_size_mb:.1f} MB")


def compute_scene_metadata(splats: dict, total_frames: int) -> dict:
    """Compute scene metadata from checkpoint."""
    means = splats["means"].cpu().numpy()
    N = means.shape[0]

    # Scene center (percentile-based, same as viewer_4d.py)
    min_coords = np.percentile(means, 5, axis=0)
    max_coords = np.percentile(means, 95, axis=0)
    center = ((min_coords + max_coords) / 2).tolist()

    # Scene radius
    distances = np.linalg.norm(means - np.array(center), axis=1)
    radius = float(np.percentile(distances, 95))

    return {
        "center": center,
        "radius": radius,
        "numSplats": N,
        "totalFrames": total_frames,
    }


def generate_synthetic_camera_presets(
    splats: dict, n_cams: int = 12, fov: float = 50.0
) -> tuple:
    """
    Generate synthetic orbit camera presets from checkpoint data.

    Use this when the original COLMAP data is not available (e.g., checkpoint
    trained on a remote server with different data paths).

    Cameras are placed on a sphere around the scene center, looking inward.
    The "up" direction is estimated via PCA of the Gaussian means.

    Returns (presets, lookatCenter).
    """
    means = splats["means"].cpu().numpy().astype(np.float64)

    # Scene center (percentile-based, robust to outliers)
    p5 = np.percentile(means, 5, axis=0)
    p95 = np.percentile(means, 95, axis=0)
    center = (p5 + p95) / 2

    # Camera distance: 95th percentile radius * 2.5
    distances = np.linalg.norm(means - center, axis=1)
    radius = float(np.percentile(distances, 95))
    cam_dist = radius * 2.5

    # Estimate "up" direction via PCA on means
    # The axis with least variance is typically the "up" direction
    centered = means - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Smallest eigenvalue → principal axis with least spread = likely vertical
    up_axis = eigenvectors[:, 0]  # smallest eigenvalue is first for eigh
    # Make up point "upward" (positive y convention if ambiguous)
    if up_axis[1] < 0:
        up_axis = -up_axis
    up_axis = up_axis / np.linalg.norm(up_axis)

    print(f"  Scene center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Scene radius: {radius:.3f}")
    print(f"  Camera distance: {cam_dist:.3f}")
    print(f"  Estimated up: ({up_axis[0]:.3f}, {up_axis[1]:.3f}, {up_axis[2]:.3f})")

    # Build a local coordinate frame around "up"
    # Pick a forward direction not parallel to up
    if abs(np.dot(up_axis, np.array([1, 0, 0]))) < 0.9:
        fwd = np.cross(up_axis, np.array([1, 0, 0]))
    else:
        fwd = np.cross(up_axis, np.array([0, 0, 1]))
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up_axis)
    right = right / np.linalg.norm(right)

    presets = []
    for i in range(n_cams):
        angle = 2 * np.pi * i / n_cams
        # Slightly elevated (30 degrees above equator)
        elev = np.radians(25)

        # Camera position on sphere
        cos_e = np.cos(elev)
        sin_e = np.sin(elev)
        pos = center + cam_dist * (
            cos_e * np.cos(angle) * right +
            cos_e * np.sin(angle) * fwd +
            sin_e * up_axis
        )

        # Camera orientation: look at center
        z_cam = center - pos  # forward = toward center
        z_cam = z_cam / np.linalg.norm(z_cam)
        x_cam = np.cross(z_cam, up_axis)
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.cross(z_cam, fwd)
        x_cam = x_cam / np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)

        # Flip Y and Z for viewer convention (same as extract_camera_presets)
        M = np.column_stack([x_cam, -y_cam, -z_cam])

        # Column-major 4x4 worldMatrix
        mat4 = [
            float(M[0, 0]), float(M[1, 0]), float(M[2, 0]), 0.0,
            float(M[0, 1]), float(M[1, 1]), float(M[2, 1]), 0.0,
            float(M[0, 2]), float(M[1, 2]), float(M[2, 2]), 0.0,
            float(pos[0]),  float(pos[1]),  float(pos[2]),  1.0,
        ]

        presets.append({
            "name": f"orbit_{i:02d}",
            "position": pos.tolist(),
            "worldMatrix": mat4,
            "fov": round(fov, 2),
        })

    lookat_center = center.tolist()
    print(f"  Generated {len(presets)} synthetic orbit camera presets")

    return presets, lookat_center


def extract_camera_presets(colmap_dir: str, max_cams: int = 8) -> tuple:
    """
    Extract normalized camera presets from COLMAP.

    CRITICAL: Must apply the EXACT same normalization as FreeTimeParser
    in training (normalize=True). The training code uses:
      T1 = similarity_from_cameras(camtoworlds)
      T2 = align_principle_axes(COLMAP_3D_points)  ← NOT camera positions!

    Returns (presets, lookatCenter) — same format as extract_camera_presets.py.
    """
    from datasets.read_write_model import (
        read_images_binary, read_cameras_binary, read_points3D_binary,
    )
    from datasets.normalize import (
        similarity_from_cameras,
        transform_cameras,
        transform_points,
        align_principle_axes,
    )

    images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
    cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))

    # Load COLMAP 3D points (needed for PCA alignment matching training)
    points3D_dict = read_points3D_binary(os.path.join(colmap_dir, "points3D.bin"))
    scene_points = np.array([p.xyz for p in points3D_dict.values()], dtype=np.float64)
    print(f"  Loaded {len(scene_points)} COLMAP 3D points for normalization")

    sorted_imgs = sorted(images.values(), key=lambda img: img.name)
    print(f"  Found {len(sorted_imgs)} COLMAP images")

    # Build cam-to-world matrices (same as FreeTimeParser)
    camtoworlds = []
    for img in sorted_imgs:
        w, x, y, z = img.qvec
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ])
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ img.tvec
        camtoworlds.append(c2w)
    camtoworlds = np.array(camtoworlds)

    # Apply the SAME normalization as FreeTimeParser (normalize=True):
    # Step 1: similarity_from_cameras → apply to both cameras AND scene points
    T1 = similarity_from_cameras(camtoworlds)
    camtoworlds = transform_cameras(T1, camtoworlds)
    scene_points = transform_points(T1, scene_points)

    # Step 2: PCA alignment using SCENE POINTS (NOT camera positions!)
    # This is the critical difference — training uses scene points for PCA.
    T2 = align_principle_axes(scene_points)
    camtoworlds = transform_cameras(T2, camtoworlds)

    # Evenly sample cameras
    n = len(sorted_imgs)
    indices = np.linspace(0, n - 1, min(n, max_cams), dtype=int)

    presets = []
    for idx in indices:
        img = sorted_imgs[idx]
        c2w = camtoworlds[idx]
        C = c2w[:3, 3]
        R_c2w = c2w[:3, :3]

        # Flip Y and Z for PlayCanvas / simple-viewer convention
        flip = np.diag([1.0, -1.0, -1.0])
        M = R_c2w @ flip

        # Column-major 4x4 worldMatrix
        mat4 = [
            float(M[0, 0]), float(M[1, 0]), float(M[2, 0]), 0.0,
            float(M[0, 1]), float(M[1, 1]), float(M[2, 1]), 0.0,
            float(M[0, 2]), float(M[1, 2]), float(M[2, 2]), 0.0,
            float(C[0]),    float(C[1]),    float(C[2]),    1.0,
        ]

        cam = cameras.get(img.camera_id)
        fov_y = 60.0
        if cam is not None:
            f = cam.params[0]
            h = cam.height
            fov_y = float(2 * np.degrees(np.arctan(h / (2 * f))))

        name = os.path.splitext(img.name)[0]
        presets.append({
            "name": f"cam_{name}",
            "position": C.tolist(),
            "worldMatrix": mat4,
            "fov": round(fov_y, 2),
        })

    # Look-at center (least-squares intersection of camera forward rays)
    A_mat = np.zeros((3, 3))
    b_vec = np.zeros(3)
    for c2w in camtoworlds:
        C = c2w[:3, 3]
        fwd = c2w[:3, 2]
        fwd = fwd / np.linalg.norm(fwd)
        I_ff = np.eye(3) - np.outer(fwd, fwd)
        A_mat += I_ff
        b_vec += I_ff @ C
    lookat_center = np.linalg.solve(A_mat, b_vec).tolist()

    print(f"  Extracted {len(presets)} camera presets")
    print(f"  Look-at center: ({lookat_center[0]:.3f}, {lookat_center[1]:.3f}, {lookat_center[2]:.3f})")

    return presets, lookat_center


def main():
    parser = argparse.ArgumentParser(
        description="Export checkpoint to viewer-compatible files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory (e.g., simple-viewer/public/data)")
    parser.add_argument("--total-frames", type=int, default=None,
                        help="Total frames (default: auto-detect from checkpoint or use 60)")
    parser.add_argument("--colmap-dir", type=str, default=None,
                        help="COLMAP sparse dir for camera presets (e.g., dataset/dance/colmap/sparse/0)")
    parser.add_argument("--max-cams", type=int, default=8,
                        help="Max camera presets to extract")
    parser.add_argument("--format", type=str, default="ply",
                        choices=["ply", "ftgs", "both"],
                        help="Output format: ply (PLY+params_4d), ftgs (all-in-one), both")
    parser.add_argument("--max-scale", type=float, default=0.3,
                        help="Cap 3D scales (linear) to this value. "
                             "Removes extreme background splats that cause spike artifacts "
                             "in quad-based viewers. Default 0.3 ≈ p99 of typical checkpoints. "
                             "Set to 0 to disable.")
    parser.add_argument("--max-aspect-ratio", type=float, default=20.0,
                        help="Prune splats with aspect ratio (max_scale/min_scale) exceeding "
                             "this value. Needle-like Gaussians cause streak artifacts in "
                             "quad-based viewers. Default 20. Set to 0 to disable.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    splats = ckpt["splats"]

    N = splats["means"].shape[0]
    K = splats["shN"].shape[1]
    sh_degree = {0: 0, 3: 1, 8: 2, 15: 3}.get(K, 3)

    print(f"  Splats: {N:,}")
    print(f"  SH degree: {sh_degree} (K={K})")
    print(f"  Times range: [{splats['times'].min().item():.3f}, {splats['times'].max().item():.3f}]")
    print(f"  Velocity magnitude: [{splats['velocities'].norm(dim=1).min().item():.3f}, {splats['velocities'].norm(dim=1).max().item():.3f}]")

    # --- Scale capping (removes spike artifacts in quad-based viewers) ---
    if args.max_scale > 0:
        max_log = np.log(args.max_scale)
        scales_log = splats["scales"]  # [N, 3] log-space
        scales_lin = torch.exp(scales_log)
        max_per_gauss = scales_lin.max(dim=1)[0]  # [N]
        n_capped = (max_per_gauss > args.max_scale).sum().item()
        pct = 100 * n_capped / N

        # Clamp log-space scales
        splats["scales"] = torch.clamp(scales_log, max=max_log)

        # Also prune extremely degenerate splats (max_scale > 10x threshold)
        # These are so extreme that capping still leaves artifacts
        prune_thresh = args.max_scale * 10
        prune_mask = max_per_gauss <= prune_thresh
        n_pruned = N - prune_mask.sum().item()
        if n_pruned > 0:
            for key in splats:
                splats[key] = splats[key][prune_mask]
            N = splats["means"].shape[0]
            print(f"  Scale capping: clamped {n_capped} ({pct:.2f}%) to max={args.max_scale}")
            print(f"  Pruned {n_pruned} extremely degenerate splats (scale > {prune_thresh:.1f})")
            print(f"  Remaining splats: {N:,}")
        else:
            print(f"  Scale capping: clamped {n_capped} ({pct:.2f}%) to max={args.max_scale}")

    # --- Aspect ratio filtering (removes needle-like streak artifacts) ---
    if args.max_aspect_ratio > 0:
        scales_lin = torch.exp(splats["scales"])  # [N, 3] linear
        max_s = scales_lin.max(dim=1)[0]
        min_s = scales_lin.min(dim=1)[0]
        aspect = max_s / (min_s + 1e-12)

        ar_mask = aspect <= args.max_aspect_ratio
        n_ar_pruned = N - ar_mask.sum().item()
        pct_ar = 100 * n_ar_pruned / N if N > 0 else 0

        if n_ar_pruned > 0:
            for key in splats:
                splats[key] = splats[key][ar_mask]
            N = splats["means"].shape[0]
            print(f"  Aspect ratio filtering: pruned {n_ar_pruned} ({pct_ar:.1f}%) with ratio > {args.max_aspect_ratio}")
            print(f"  Remaining splats: {N:,}")
        else:
            print(f"  Aspect ratio filtering: all splats within ratio <= {args.max_aspect_ratio}")

    # Determine total frames
    total_frames = args.total_frames
    if total_frames is None:
        # Try to get from checkpoint config
        if "cfg" in ckpt:
            cfg = ckpt["cfg"]
            if hasattr(cfg, "end_frame") and hasattr(cfg, "start_frame"):
                total_frames = cfg.end_frame - cfg.start_frame
                print(f"  Auto-detected total_frames={total_frames} from checkpoint config")
        if total_frames is None:
            total_frames = 60
            print(f"  Using default total_frames={total_frames}")

    # Export PLY + params_4d.bin
    if args.format in ("ply", "both"):
        print("\nExporting PLY + params_4d.bin format:")
        ply_path = os.path.join(args.output, "canonical.ply")
        p4d_path = os.path.join(args.output, "params_4d.bin")

        write_canonical_ply_fast(ply_path, splats)
        write_params_4d_bin(p4d_path, splats, total_frames)

    # Export FTGS
    if args.format in ("ftgs", "both"):
        print("\nExporting FTGS format:")
        ftgs_path = os.path.join(args.output, "checkpoint.ftgs")
        write_ftgs(ftgs_path, splats, total_frames)

    # Scene metadata
    print("\nComputing scene metadata:")
    meta = compute_scene_metadata(splats, total_frames)

    # Camera presets
    if args.colmap_dir:
        print("\nExtracting camera presets from COLMAP:")
        try:
            presets, lookat_center = extract_camera_presets(args.colmap_dir, args.max_cams)
            meta["cameraPresets"] = presets
            meta["lookatCenter"] = lookat_center
        except Exception as e:
            print(f"  WARNING: Failed to extract camera presets: {e}")
            print("  Falling back to synthetic camera presets...")
            presets, lookat_center = generate_synthetic_camera_presets(splats, n_cams=12)
            meta["cameraPresets"] = presets
            meta["lookatCenter"] = lookat_center
    else:
        print("\nGenerating synthetic camera presets (no COLMAP dir specified):")
        presets, lookat_center = generate_synthetic_camera_presets(splats, n_cams=12)
        meta["cameraPresets"] = presets
        meta["lookatCenter"] = lookat_center

    meta_path = os.path.join(args.output, "scene_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  scene_meta.json: {meta_path}")

    # Verification summary
    print("\n=== Export Summary ===")
    print(f"  Output directory: {args.output}")
    print(f"  Splats: {N:,}")
    print(f"  Total frames: {total_frames}")
    print(f"  SH degree: {sh_degree}")
    if args.format in ("ply", "both"):
        print(f"  canonical.ply: {os.path.getsize(os.path.join(args.output, 'canonical.ply')) / (1024*1024):.1f} MB")
        print(f"  params_4d.bin: {os.path.getsize(os.path.join(args.output, 'params_4d.bin')) / (1024*1024):.1f} MB")
    if args.format in ("ftgs", "both"):
        print(f"  checkpoint.ftgs: {os.path.getsize(os.path.join(args.output, 'checkpoint.ftgs')) / (1024*1024):.1f} MB")
    if "cameraPresets" in meta:
        print(f"  Camera presets: {len(meta['cameraPresets'])}")

    # Verification: print first few values for sanity check
    print("\n=== Verification (first splat) ===")
    print(f"  Position:   [{splats['means'][0, 0]:.6f}, {splats['means'][0, 1]:.6f}, {splats['means'][0, 2]:.6f}]")
    print(f"  Scale (log): [{splats['scales'][0, 0]:.6f}, {splats['scales'][0, 1]:.6f}, {splats['scales'][0, 2]:.6f}]")
    print(f"  Quat (wxyz): [{splats['quats'][0, 0]:.6f}, {splats['quats'][0, 1]:.6f}, {splats['quats'][0, 2]:.6f}, {splats['quats'][0, 3]:.6f}]")
    print(f"  Opacity (logit): {splats['opacities'][0]:.6f}")
    print(f"  SH DC:      [{splats['sh0'][0, 0, 0]:.6f}, {splats['sh0'][0, 0, 1]:.6f}, {splats['sh0'][0, 0, 2]:.6f}]")
    print(f"  Velocity:   [{splats['velocities'][0, 0]:.6f}, {splats['velocities'][0, 1]:.6f}, {splats['velocities'][0, 2]:.6f}]")
    print(f"  Time:       {splats['times'][0, 0]:.6f}")
    dur = torch.exp(splats["durations"][0, 0]).clamp(min=0.02)
    print(f"  Duration (activated): {dur:.6f}")


if __name__ == "__main__":
    main()
