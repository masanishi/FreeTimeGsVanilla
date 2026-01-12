#!/usr/bin/env python3
"""
Test initial point cloud rendering against ground truth.
Renders 50 random camera views and saves comparison images.
"""

import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import random

sys.path.insert(0, str(Path(__file__).parent))

from gsplat import rasterization
from datasets.FreeTime_dataset import FreeTimeParser, load_multiframe_colmap_points
from utils import rgb_to_sh, knn


def test_initial_rendering(
    data_dir: str,
    output_dir: str = "test_renders",
    num_views: int = 50,
    start_frame: int = 0,
    end_frame: int = 51,
    frame_step: int = 10,
):
    """Render initial point cloud from multiple camera views."""

    device = torch.device("cuda:0")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    # Load parser
    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    print(f"Scene scale: {parser.scene_scale}")
    print(f"Number of cameras: {len(parser.camtoworlds)}")

    # Load multiframe points
    print("\nLoading multiframe COLMAP points...")
    init_data = load_multiframe_colmap_points(
        data_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        max_error=1.0,
        match_threshold=0.1,
        transform=parser.transform
    )

    positions = init_data['positions']
    colors = init_data['colors']
    times = init_data['times']
    velocities = init_data['velocities']

    print(f"Total points: {len(positions)}")
    print(f"Time range: [{times.min():.3f}, {times.max():.3f}]")

    # Subsample if too many points
    max_points = 500000
    if len(positions) > max_points:
        idx = torch.randperm(len(positions))[:max_points]
        positions = positions[idx]
        colors = colors[idx]
        times = times[idx]
        velocities = velocities[idx]
        print(f"Subsampled to {max_points} points")

    N = len(positions)

    # Initialize Gaussian parameters (same as trainer)
    print("\n" + "=" * 60)
    print("INITIALIZING GAUSSIANS")
    print("=" * 60)

    init_scale = 0.5
    init_duration = 5.0  # Large duration so ALL points visible at ALL times initially
    init_opacity = 0.1  # Lower opacity to reduce color averaging

    # Scales from KNN
    dist2_avg = (knn(positions, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    dist_avg = torch.clamp(dist_avg, min=1e-6)
    scales = torch.exp(torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3))

    print(f"Scales: min={scales.min():.6f}, max={scales.max():.6f}, mean={scales.mean():.6f}")

    # Quats (identity)
    quats = torch.zeros(N, 4)
    quats[:, 0] = 1.0

    # Opacities
    base_opacity = torch.full((N,), init_opacity)

    # Duration
    durations = torch.full((N, 1), init_duration)

    # SH colors
    sh0 = rgb_to_sh(colors).unsqueeze(1)  # [N, 1, 3]

    print(f"Base opacity: {init_opacity}")
    print(f"Duration: {init_duration}")
    print(f"SH colors: min={sh0.min():.3f}, max={sh0.max():.3f}")

    # Move to device
    means = positions.to(device)
    quats = quats.to(device)
    scales = scales.to(device)
    base_opacity = base_opacity.to(device)
    mu_t = times.to(device)
    durations = durations.to(device)
    velocities = velocities.to(device)
    sh0 = sh0.to(device)

    # Select random cameras
    num_cameras = len(parser.camtoworlds)
    camera_indices = random.sample(range(num_cameras), min(num_views, num_cameras))

    # Also select random times
    test_times = np.linspace(0, 1, num_views)

    print("\n" + "=" * 60)
    print(f"RENDERING {len(camera_indices)} VIEWS")
    print("=" * 60)

    psnr_values = []

    for i, cam_idx in enumerate(camera_indices):
        t = test_times[i % len(test_times)]

        # Get camera info
        camtoworld = torch.from_numpy(parser.camtoworlds[cam_idx]).float().unsqueeze(0).to(device)
        camera_id = parser.camera_ids[cam_idx]
        K = torch.from_numpy(parser.Ks_dict[camera_id]).float().unsqueeze(0).to(device)
        width, height = parser.imsize_dict[camera_id]

        viewmat = torch.linalg.inv(camtoworld)

        # Compute moved positions: pos(t) = pos_init + vel * (t - mu_t)
        moved_means = means + velocities * (t - mu_t)

        # Compute temporal opacity: exp(-0.5 * ((t - mu_t) / duration)^2)
        temporal_op = torch.exp(-0.5 * ((t - mu_t) / (durations + 1e-8)) ** 2).squeeze()
        effective_op = base_opacity * temporal_op

        # Render
        render_colors, render_alphas, _ = rasterization(
            means=moved_means,
            quats=quats,
            scales=scales,
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

        # Load ground truth image
        frame_num = int(t * (end_frame - start_frame)) + start_frame
        frame_num = min(frame_num, end_frame - 1)

        # Construct image path manually
        frame_with_offset = frame_num + parser.frame_start_offset
        frame_name = f"{frame_with_offset:0{parser.frame_digits}d}{parser.image_format}"
        gt_path = Path(parser.campaths[cam_idx]) / frame_name

        if gt_path.exists():
            gt_img = np.array(Image.open(gt_path))[:, :, :3] / 255.0
        else:
            # Try frame 0
            frame_name = f"{parser.frame_start_offset:0{parser.frame_digits}d}{parser.image_format}"
            gt_path = Path(parser.campaths[cam_idx]) / frame_name
            if gt_path.exists():
                gt_img = np.array(Image.open(gt_path))[:, :, :3] / 255.0
            else:
                gt_img = np.zeros((height, width, 3))

        # Process rendered image
        render_np = torch.clamp(render_colors, 0, 1).squeeze(0).cpu().numpy()
        alpha_np = render_alphas.squeeze(0).cpu().numpy()

        # Compute PSNR
        mse = np.mean((render_np - gt_img) ** 2)
        if mse > 0:
            psnr = -10 * np.log10(mse)
        else:
            psnr = 100
        psnr_values.append(psnr)

        # Create comparison image: GT | Rendered | Alpha
        gt_uint8 = (gt_img * 255).astype(np.uint8)
        render_uint8 = (render_np * 255).astype(np.uint8)
        alpha_uint8 = (np.stack([alpha_np.squeeze()] * 3, axis=-1) * 255).astype(np.uint8)

        comparison = np.hstack([gt_uint8, render_uint8, alpha_uint8])

        # Save
        save_path = output_path / f"view_{i:03d}_cam{cam_idx}_t{t:.2f}.png"
        cv2.imwrite(str(save_path), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        print(f"View {i+1}/{len(camera_indices)}: cam={cam_idx}, t={t:.2f}, "
              f"PSNR={psnr:.2f}, render=[{render_np.min():.3f}, {render_np.max():.3f}], "
              f"alpha=[{alpha_np.min():.3f}, {alpha_np.max():.3f}]")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
    print(f"Min PSNR: {np.min(psnr_values):.2f} dB")
    print(f"Max PSNR: {np.max(psnr_values):.2f} dB")
    print(f"\nSaved {len(camera_indices)} comparison images to: {output_path}")
    print("Layout: [Ground Truth | Rendered | Alpha]")

    # Create a grid summary
    print("\nCreating summary grid...")
    grid_size = min(25, len(camera_indices))
    grid_images = []

    for i in range(grid_size):
        img_path = output_path / f"view_{i:03d}_cam{camera_indices[i]}_t{test_times[i]:.2f}.png"
        if img_path.exists():
            img = cv2.imread(str(img_path))
            # Resize for grid
            img = cv2.resize(img, (600, 200))
            grid_images.append(img)

    if grid_images:
        # Create 5x5 grid (or smaller)
        rows = 5
        cols = 5
        grid_h = rows * 200
        grid_w = cols * 600
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for idx, img in enumerate(grid_images[:25]):
            r = idx // cols
            c = idx % cols
            grid[r*200:(r+1)*200, c*600:(c+1)*600] = img

        grid_path = output_path / "summary_grid.png"
        cv2.imwrite(str(grid_path), grid)
        print(f"Saved summary grid to: {grid_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="test_renders")
    parser.add_argument("--num-views", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=51)
    parser.add_argument("--frame-step", type=int, default=10)
    args = parser.parse_args()

    test_initial_rendering(
        args.data_dir,
        args.output_dir,
        args.num_views,
        args.start_frame,
        args.end_frame,
        args.frame_step,
    )
