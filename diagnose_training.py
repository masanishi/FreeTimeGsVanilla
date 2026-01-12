#!/usr/bin/env python3
"""
Diagnose training issues by monitoring key statistics.
"""

import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from gsplat import rasterization
from datasets.FreeTime_dataset import FreeTimeParser, load_multiframe_colmap_points
from utils import rgb_to_sh, knn


def diagnose(data_dir: str):
    """Run diagnostics on training setup."""

    device = torch.device("cuda:0")

    # Load parser
    print("="*60)
    print("LOADING DATA")
    print("="*60)

    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    print(f"Scene scale: {parser.scene_scale}")
    print(f"Transform:\n{parser.transform}")

    # Check camera matrices
    print("\n" + "="*60)
    print("CAMERA MATRIX CHECK")
    print("="*60)

    for i in range(min(3, len(parser.camtoworlds))):
        c2w = parser.camtoworlds[i]
        R = c2w[:3, :3]
        row_norms = np.linalg.norm(R, axis=1)
        det = np.linalg.det(R)
        orth = np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        print(f"Camera {i}: row_norms={row_norms}, det={det:.6f}, orthogonal={orth}")

        if not orth:
            print(f"  WARNING: Camera matrix not orthonormal!")

    # Load multiframe points
    print("\n" + "="*60)
    print("POINT CLOUD CHECK")
    print("="*60)

    init_data = load_multiframe_colmap_points(
        data_dir,
        start_frame=0,
        end_frame=60,
        frame_step=10,
        max_error=1.0,
        match_threshold=0.1,
        transform=parser.transform
    )

    positions = init_data['positions']
    colors = init_data['colors']
    times = init_data['times']
    velocities = init_data['velocities']
    has_velocity = init_data['has_velocity']

    print(f"Total points: {len(positions)}")
    print(f"Position range: X=[{positions[:,0].min():.3f}, {positions[:,0].max():.3f}]")
    print(f"Position range: Y=[{positions[:,1].min():.3f}, {positions[:,1].max():.3f}]")
    print(f"Position range: Z=[{positions[:,2].min():.3f}, {positions[:,2].max():.3f}]")
    print(f"Time range: [{times.min():.3f}, {times.max():.3f}]")
    print(f"Has velocity: {has_velocity.sum()}/{len(has_velocity)} ({100*has_velocity.float().mean():.1f}%)")

    vel_mags = velocities.norm(dim=1)
    print(f"Velocity magnitude: min={vel_mags.min():.6f}, max={vel_mags.max():.6f}, mean={vel_mags.mean():.6f}")

    # Subsample for testing
    if len(positions) > 100000:
        idx = torch.randperm(len(positions))[:100000]
        positions = positions[idx]
        colors = colors[idx]
        times = times[idx]
        velocities = velocities[idx]

    N = len(positions)

    # Initialize Gaussian parameters (same as trainer)
    print("\n" + "="*60)
    print("GAUSSIAN INITIALIZATION")
    print("="*60)

    init_scale = 0.3
    init_duration = 5.0  # Large duration so ALL points visible at ALL times initially
    init_opacity = 0.5

    # Scales from KNN
    dist2_avg = (knn(positions, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    dist_avg = torch.clamp(dist_avg, min=1e-6)
    scales_log = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)
    scales = torch.exp(scales_log)

    print(f"Scales (log): min={scales_log.min():.3f}, max={scales_log.max():.3f}, mean={scales_log.mean():.3f}")
    print(f"Scales: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")

    # Quats (identity)
    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0

    # Opacities
    opacities_logit = torch.logit(torch.full((N,), init_opacity))
    base_opacity = torch.sigmoid(opacities_logit)

    print(f"Base opacity: min={base_opacity.min():.4f}, max={base_opacity.max():.4f}, mean={base_opacity.mean():.4f}")

    # Duration
    durations_log = torch.log(torch.full((N, 1), init_duration))
    durations = torch.exp(durations_log)

    print(f"Duration: {durations[0].item():.3f}")

    # Temporal opacity at different times
    print("\n" + "="*60)
    print("TEMPORAL OPACITY CHECK")
    print("="*60)

    mu_t = times  # [N, 1]
    s = durations  # [N, 1]

    for query_t in [0.0, 0.2, 0.5, 0.8, 1.0]:
        temporal_op = torch.exp(-0.5 * ((query_t - mu_t) / (s + 1e-8)) ** 2).squeeze()
        effective_op = base_opacity * temporal_op

        print(f"\nAt t={query_t}:")
        print(f"  Temporal opacity: min={temporal_op.min():.6f}, max={temporal_op.max():.6f}, mean={temporal_op.mean():.6f}")
        print(f"  Effective opacity: min={effective_op.min():.6f}, max={effective_op.max():.6f}, mean={effective_op.mean():.6f}")

        # Count visible Gaussians (effective opacity > 0.01)
        n_visible = (effective_op > 0.01).sum().item()
        print(f"  Visible Gaussians (eff_op > 0.01): {n_visible}/{N} ({100*n_visible/N:.1f}%)")

    # Test rendering at different times
    print("\n" + "="*60)
    print("RENDER TEST")
    print("="*60)

    # Move to device
    means = positions.to(device)
    quats_d = quats.to(device)
    scales_d = scales.to(device)
    base_op_d = base_opacity.to(device)
    mu_t_d = mu_t.to(device)
    durations_d = durations.to(device)
    velocities_d = velocities.to(device)

    # SH colors
    sh0 = rgb_to_sh(colors).unsqueeze(1).to(device)  # [N, 1, 3]

    # Get camera
    cam_idx = 0
    camtoworld = torch.from_numpy(parser.camtoworlds[cam_idx]).float().unsqueeze(0).to(device)
    camera_id = parser.camera_ids[cam_idx]
    K = torch.from_numpy(parser.Ks_dict[camera_id]).float().unsqueeze(0).to(device)
    width, height = parser.imsize_dict[camera_id]

    viewmat = torch.linalg.inv(camtoworld)

    print(f"Camera position: {camtoworld[0, :3, 3].cpu().numpy()}")
    print(f"Image size: {width}x{height}")

    for query_t in [0.0, 0.2, 0.5, 0.8, 1.0]:
        # Compute moved positions
        moved_means = means + velocities_d * (query_t - mu_t_d)

        # Compute temporal opacity
        temporal_op = torch.exp(-0.5 * ((query_t - mu_t_d) / (durations_d + 1e-8)) ** 2).squeeze()
        effective_op = base_op_d * temporal_op

        # Render
        render_colors, render_alphas, info = rasterization(
            means=moved_means,
            quats=quats_d,
            scales=scales_d,
            opacities=effective_op,
            colors=sh0,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=0,
            near_plane=0.01,
            far_plane=100.0,
        )

        print(f"\nAt t={query_t}:")
        print(f"  Render colors: min={render_colors.min():.4f}, max={render_colors.max():.4f}, mean={render_colors.mean():.4f}")
        print(f"  Render alphas: min={render_alphas.min():.4f}, max={render_alphas.max():.4f}, mean={render_alphas.mean():.4f}")

        # Save first and last
        if query_t in [0.0, 1.0]:
            import cv2
            render_np = torch.clamp(render_colors, 0, 1).squeeze(0).cpu().numpy()
            cv2.imwrite(f"diag_render_t{query_t:.1f}.png",
                       cv2.cvtColor((render_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            print(f"  Saved: diag_render_t{query_t:.1f}.png")

    # Check: What if all Gaussians have time=0.5 and large duration?
    print("\n" + "="*60)
    print("SANITY CHECK: All Gaussians at t=0.5, duration=1.0")
    print("="*60)

    mu_t_center = torch.full((N, 1), 0.5, device=device)
    dur_large = torch.full((N, 1), 1.0, device=device)

    for query_t in [0.0, 0.5, 1.0]:
        temporal_op = torch.exp(-0.5 * ((query_t - mu_t_center) / (dur_large + 1e-8)) ** 2).squeeze()
        effective_op = base_op_d * temporal_op

        render_colors, render_alphas, info = rasterization(
            means=means,  # Static positions
            quats=quats_d,
            scales=scales_d,
            opacities=effective_op,
            colors=sh0,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=0,
            near_plane=0.01,
            far_plane=100.0,
        )

        print(f"At t={query_t}: colors=[{render_colors.min():.4f}, {render_colors.max():.4f}], alpha=[{render_alphas.min():.4f}, {render_alphas.max():.4f}]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    diagnose(args.data_dir)
