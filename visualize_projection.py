#!/usr/bin/env python3
"""
Visualize point cloud projection onto images to verify coordinate system.
This will help diagnose if points are correctly aligned with cameras.
"""

import sys
import os
import numpy as np
from pathlib import Path
import torch
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from datasets.FreeTime_dataset import FreeTimeParser, load_multiframe_colmap_points
from read_write_model import read_model, qvec2rotmat

def project_points_to_image(points_3d, K, c2w):
    """
    Project 3D points to image coordinates.

    Args:
        points_3d: [N, 3] points in world coordinates
        K: [3, 3] intrinsic matrix
        c2w: [4, 4] camera-to-world matrix

    Returns:
        uv: [N, 2] pixel coordinates
        depths: [N] depth values
    """
    # World to camera: w2c = inv(c2w)
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3]
    t = w2c[:3, 3]

    # Transform to camera coords
    points_cam = (R @ points_3d.T).T + t  # [N, 3]

    # Get depths (z coordinate in camera space)
    depths = points_cam[:, 2]

    # Project to image (avoid division by zero)
    valid_depth = depths > 0.01
    uv = np.zeros((len(points_3d), 2))

    # Only project points in front of camera
    if valid_depth.any():
        pts_valid = points_cam[valid_depth]
        uv_valid = (K @ pts_valid.T).T
        uv_valid = uv_valid[:, :2] / uv_valid[:, 2:3]
        uv[valid_depth] = uv_valid

    return uv, depths

def visualize_projection(data_dir, output_path="projection_test.png",
                         camera_idx=0, frame_idx=0, max_points=50000):
    """
    Project RoMa points onto an image and visualize.
    """
    print(f"\n{'='*60}")
    print("Loading parser and points...")
    print(f"{'='*60}")

    # Load parser
    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    print(f"Scene scale: {parser.scene_scale}")
    print(f"Transform:\n{parser.transform}")

    # Load multiframe points
    print("\nLoading multiframe COLMAP points...")
    init_data = load_multiframe_colmap_points(
        data_dir,
        start_frame=0,
        end_frame=60,  # Just a few frames
        frame_step=10,
        max_error=2.0,
        match_threshold=0.1,
        transform=parser.transform
    )

    positions = init_data['positions'].numpy()
    colors = init_data['colors'].numpy()

    print(f"\nLoaded {len(positions)} points")
    print(f"Position range X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
    print(f"Position range Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
    print(f"Position range Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

    # Sample points if too many
    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
        colors = colors[idx]
        print(f"Sampled to {max_points} points")

    # Get camera info
    camtoworld = parser.camtoworlds[camera_idx]
    camera_name = parser.camera_names[camera_idx]
    camera_id = parser.camera_ids[camera_idx]
    K = parser.Ks_dict[camera_id]
    width, height = parser.imsize_dict[camera_id]

    print(f"\nCamera: {camera_name} (id={camera_id})")
    print(f"Image size: {width}x{height}")
    print(f"K:\n{K}")
    print(f"Camera position: {camtoworld[:3, 3]}")

    # Project points
    uv, depths = project_points_to_image(positions, K, camtoworld)

    # Filter valid projections
    valid = (depths > 0.01) & \
            (uv[:, 0] >= 0) & (uv[:, 0] < width) & \
            (uv[:, 1] >= 0) & (uv[:, 1] < height)

    n_valid = valid.sum()
    print(f"\nPoints in front of camera: {(depths > 0.01).sum()}/{len(depths)}")
    print(f"Points projecting to image: {n_valid}/{len(depths)}")

    # Load actual image
    img_path = parser._get_image_path(parser.campaths[camera_idx], frame_idx)
    print(f"\nLoading image: {img_path}")

    if Path(img_path).exists():
        img = np.array(Image.open(img_path))
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        img = img[:, :, :3]  # Remove alpha if present
    else:
        print(f"Image not found, creating blank")
        img = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw projected points
    img_with_points = img.copy()

    valid_uv = uv[valid].astype(np.int32)
    valid_colors = (colors[valid] * 255).astype(np.uint8)

    for i in range(len(valid_uv)):
        x, y = valid_uv[i]
        color = tuple(int(c) for c in valid_colors[i])
        cv2.circle(img_with_points, (x, y), 2, color, -1)

    # Save result
    cv2.imwrite(output_path, cv2.cvtColor(img_with_points, cv2.COLOR_RGB2BGR))
    print(f"\nSaved projection visualization to: {output_path}")

    # Also save side-by-side comparison
    comparison = np.hstack([img, img_with_points])
    comparison_path = output_path.replace('.png', '_comparison.png')
    cv2.imwrite(comparison_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    print(f"Saved comparison to: {comparison_path}")

    # Statistics about projection
    if n_valid > 0:
        print(f"\n--- Projection Statistics ---")
        print(f"UV X range: [{valid_uv[:, 0].min()}, {valid_uv[:, 0].max()}]")
        print(f"UV Y range: [{valid_uv[:, 1].min()}, {valid_uv[:, 1].max()}]")

        # Check depth distribution
        valid_depths = depths[valid]
        print(f"Depth range: [{valid_depths.min():.3f}, {valid_depths.max():.3f}]")
        print(f"Depth mean: {valid_depths.mean():.3f}")

    return positions, uv, depths, valid

def compare_reference_vs_roma_projection(data_dir, output_dir="debug_projections"):
    """Compare projection of reference COLMAP points vs RoMa points."""
    from pathlib import Path
    import os

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*80)
    print("COMPARING REFERENCE VS ROMA POINT PROJECTIONS")
    print("="*80)

    # Load parser (this loads reference COLMAP)
    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    # Reference points (already transformed by parser)
    ref_points = parser.points
    ref_colors = parser.points_rgb / 255.0

    print(f"\nReference COLMAP: {len(ref_points)} points")
    print(f"Reference position range X: [{ref_points[:, 0].min():.3f}, {ref_points[:, 0].max():.3f}]")
    print(f"Reference position range Y: [{ref_points[:, 1].min():.3f}, {ref_points[:, 1].max():.3f}]")
    print(f"Reference position range Z: [{ref_points[:, 2].min():.3f}, {ref_points[:, 2].max():.3f}]")

    # Load RoMa points
    roma_data = load_multiframe_colmap_points(
        data_dir,
        start_frame=0,
        end_frame=10,
        frame_step=10,
        max_error=2.0,
        match_threshold=0.1,
        transform=parser.transform
    )
    roma_points = roma_data['positions'].numpy()
    roma_colors = roma_data['colors'].numpy()

    print(f"\nRoMa COLMAP: {len(roma_points)} points")
    print(f"RoMa position range X: [{roma_points[:, 0].min():.3f}, {roma_points[:, 0].max():.3f}]")
    print(f"RoMa position range Y: [{roma_points[:, 1].min():.3f}, {roma_points[:, 1].max():.3f}]")
    print(f"RoMa position range Z: [{roma_points[:, 2].min():.3f}, {roma_points[:, 2].max():.3f}]")

    # Test on multiple cameras
    test_cameras = [0, 1, 2, 10, 20, 30]

    for cam_idx in test_cameras:
        if cam_idx >= len(parser.camera_names):
            continue

        camtoworld = parser.camtoworlds[cam_idx]
        camera_name = parser.camera_names[cam_idx]
        camera_id = parser.camera_ids[cam_idx]
        K = parser.Ks_dict[camera_id]
        width, height = parser.imsize_dict[camera_id]

        # Load image
        frame_num = 0 + parser.frame_start_offset
        frame_name = f"{frame_num:0{parser.frame_digits}d}{parser.image_format}"
        img_path = os.path.join(parser.campaths[cam_idx], frame_name)
        if Path(img_path).exists():
            img = np.array(Image.open(img_path))[:, :, :3]
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8)

        # Project reference points
        ref_uv, ref_depths = project_points_to_image(ref_points, K, camtoworld)
        ref_valid = (ref_depths > 0.01) & \
                   (ref_uv[:, 0] >= 0) & (ref_uv[:, 0] < width) & \
                   (ref_uv[:, 1] >= 0) & (ref_uv[:, 1] < height)

        # Project RoMa points
        roma_uv, roma_depths = project_points_to_image(roma_points, K, camtoworld)
        roma_valid = (roma_depths > 0.01) & \
                    (roma_uv[:, 0] >= 0) & (roma_uv[:, 0] < width) & \
                    (roma_uv[:, 1] >= 0) & (roma_uv[:, 1] < height)

        print(f"\nCamera {cam_idx} ({camera_name}):")
        print(f"  Reference: {ref_valid.sum()}/{len(ref_points)} points project to image")
        print(f"  RoMa: {roma_valid.sum()}/{len(roma_points)} points project to image")

        # Draw reference (blue)
        img_ref = img.copy()
        for i in np.where(ref_valid)[0][:10000]:  # Limit for speed
            x, y = int(ref_uv[i, 0]), int(ref_uv[i, 1])
            cv2.circle(img_ref, (x, y), 2, (0, 0, 255), -1)

        # Draw RoMa (green)
        img_roma = img.copy()
        for i in np.where(roma_valid)[0][:10000]:
            x, y = int(roma_uv[i, 0]), int(roma_uv[i, 1])
            cv2.circle(img_roma, (x, y), 2, (0, 255, 0), -1)

        # Save comparison
        comparison = np.hstack([img, img_ref, img_roma])
        cv2.imwrite(str(output_dir / f"cam{cam_idx:02d}_{camera_name}_comparison.png"),
                   cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

    print(f"\nSaved comparison images to: {output_dir}")

if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data-dir", type=str, required=True)
    argparser.add_argument("--output", type=str, default="projection_test.png")
    argparser.add_argument("--camera", type=int, default=0)
    argparser.add_argument("--frame", type=int, default=0)
    argparser.add_argument("--compare", action="store_true", help="Compare reference vs RoMa")
    args = argparser.parse_args()

    if args.compare:
        compare_reference_vs_roma_projection(args.data_dir)
    else:
        visualize_projection(args.data_dir, args.output, args.camera, args.frame)
