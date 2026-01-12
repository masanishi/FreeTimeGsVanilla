#!/usr/bin/env python3
"""
Debug rendering to understand why images are black.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from gsplat import rasterization
from datasets.FreeTime_dataset import FreeTimeParser, load_multiframe_colmap_points
from utils import rgb_to_sh

def test_render(data_dir, output_path="debug_render.png"):
    """Test basic rendering with the point cloud."""

    device = torch.device("cuda:0")

    # Load parser
    print("Loading parser...")
    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    print(f"Scene scale: {parser.scene_scale}")

    # Load multiframe points
    print("\nLoading multiframe COLMAP points...")
    init_data = load_multiframe_colmap_points(
        data_dir,
        start_frame=0,
        end_frame=10,
        frame_step=10,
        max_error=1.0,
        match_threshold=0.1,
        transform=parser.transform
    )

    points = init_data['positions']
    rgbs = init_data['colors']

    print(f"Points shape: {points.shape}")
    print(f"RGB range: [{rgbs.min():.3f}, {rgbs.max():.3f}]")

    # Sample to manageable size
    if len(points) > 50000:
        idx = torch.randperm(len(points))[:50000]
        points = points[idx]
        rgbs = rgbs[idx]

    N = len(points)
    print(f"Using {N} points")

    # Initialize Gaussian parameters
    means = points.to(device)

    # Scales from KNN (matching trainer settings)
    init_scale = 0.3  # Same as trainer default
    from utils import knn
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    dist_avg = torch.clamp(dist_avg, min=1e-6)  # Prevent log(0)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3).to(device)
    print(f"Scales (log): min={scales.min():.3f}, max={scales.max():.3f}")

    # Quats (identity)
    quats = torch.zeros(N, 4, device=device)
    quats[:, 0] = 1.0

    # Opacities
    opacities = torch.ones(N, device=device) * 0.5

    # Colors - SH DC only (sh_degree=0)
    # Test 1: Use raw RGB directly (no SH)
    colors_raw = rgbs.to(device)

    # Test 2: Use SH-converted colors
    sh0 = rgb_to_sh(rgbs).unsqueeze(1)  # [N, 1, 3]
    colors_sh = sh0.to(device)

    print(f"\nRaw RGB colors: min={colors_raw.min():.3f}, max={colors_raw.max():.3f}")
    print(f"SH coefficients: min={colors_sh.min():.3f}, max={colors_sh.max():.3f}")

    # Get camera info
    cam_idx = 0
    camtoworld = torch.from_numpy(parser.camtoworlds[cam_idx]).float().unsqueeze(0).to(device)
    camera_id = parser.camera_ids[cam_idx]
    K = torch.from_numpy(parser.Ks_dict[camera_id]).float().unsqueeze(0).to(device)
    width, height = parser.imsize_dict[camera_id]

    viewmat = torch.linalg.inv(camtoworld)

    print(f"\nCamera {cam_idx}:")
    print(f"  Image size: {width}x{height}")
    print(f"  K:\n{K[0]}")
    print(f"  Camera position: {camtoworld[0, :3, 3]}")

    # Test render with SH colors (sh_degree=0)
    print("\n--- Rendering with SH colors (sh_degree=0) ---")
    render_colors_sh, render_alphas_sh, info_sh = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=opacities,
        colors=colors_sh,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=0,
        near_plane=0.01,
        far_plane=100.0,
    )

    print(f"Render colors (SH): min={render_colors_sh.min():.4f}, max={render_colors_sh.max():.4f}, mean={render_colors_sh.mean():.4f}")
    print(f"Render alphas: min={render_alphas_sh.min():.4f}, max={render_alphas_sh.max():.4f}")

    # Test render with raw RGB colors (sh_degree=None to skip SH eval)
    # gsplat doesn't support this, so let's check what SH eval does

    # Manual SH evaluation check
    C0 = 0.28209479177387814
    # SH to RGB: color = sh0 * C0 + 0.5
    sh_back_to_rgb = colors_sh.squeeze(1) * C0 + 0.5
    print(f"\nSH back to RGB: min={sh_back_to_rgb.min():.4f}, max={sh_back_to_rgb.max():.4f}")
    print(f"Original RGB: min={colors_raw.min():.4f}, max={colors_raw.max():.4f}")
    print(f"Difference (should be ~0): {(sh_back_to_rgb - colors_raw).abs().max():.6f}")

    # Load actual image for comparison
    frame_num = 0 + parser.frame_start_offset
    frame_name = f"{frame_num:0{parser.frame_digits}d}{parser.image_format}"
    img_path = Path(parser.campaths[cam_idx]) / frame_name

    if img_path.exists():
        gt_image = np.array(Image.open(img_path))[:, :, :3] / 255.0
        print(f"\nGT image: min={gt_image.min():.4f}, max={gt_image.max():.4f}")
    else:
        gt_image = np.zeros((height, width, 3))
        print(f"GT image not found: {img_path}")

    # Save rendered images
    render_sh_np = torch.clamp(render_colors_sh, 0, 1).squeeze(0).cpu().numpy()
    alpha_np = render_alphas_sh.squeeze(0).cpu().numpy()

    # Create comparison
    fig_h = height
    fig_w = width * 3
    canvas = np.zeros((fig_h, fig_w, 3), dtype=np.uint8)

    # GT
    canvas[:, :width] = (gt_image * 255).astype(np.uint8)

    # Rendered (SH)
    canvas[:, width:2*width] = (render_sh_np * 255).astype(np.uint8)

    # Alpha
    alpha_rgb = np.stack([alpha_np.squeeze(-1)]*3, axis=-1)
    canvas[:, 2*width:] = (alpha_rgb * 255).astype(np.uint8)

    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"\nSaved comparison to: {output_path}")
    print("Layout: [GT | Rendered | Alpha]")

    # Additional debug: check what colors gsplat produces for a simple case
    print("\n--- Simple test: uniform white gaussians ---")
    # All white points should produce white output
    white_sh = rgb_to_sh(torch.ones(N, 3)).unsqueeze(1).to(device)
    print(f"White SH: {white_sh[0]}")

    render_white, _, _ = rasterization(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=opacities,
        colors=white_sh,
        viewmats=viewmat,
        Ks=K,
        width=width,
        height=height,
        sh_degree=0,
        near_plane=0.01,
        far_plane=100.0,
    )
    print(f"White render: min={render_white.min():.4f}, max={render_white.max():.4f}")

    # Save white render
    white_np = torch.clamp(render_white, 0, 1).squeeze(0).cpu().numpy()
    cv2.imwrite("debug_render_white.png", cv2.cvtColor((white_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print("Saved white render to: debug_render_white.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    test_render(args.data_dir)
