#!/usr/bin/env python3
"""
Debug script to verify RoMa triangulation and coordinate systems.
Generates point cloud for one frame and visualizes against camera poses.
"""

import sys
import numpy as np
from pathlib import Path
import torch

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
from read_write_model import read_model, qvec2rotmat

def get_camera_center(image):
    """Get camera center in world coordinates: C = -R.T @ t"""
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    C = -R.T @ t
    return C.flatten()

def load_colmap_points(colmap_path):
    """Load COLMAP points and cameras."""
    cameras, images, points3D = read_model(colmap_path)

    # Get camera centers
    cam_centers = []
    cam_names = []
    for img_id, img in images.items():
        center = get_camera_center(img)
        cam_centers.append(center)
        cam_names.append(img.name)
    cam_centers = np.array(cam_centers)

    # Get 3D points
    if points3D:
        points = np.array([p.xyz for p in points3D.values()])
        colors = np.array([p.rgb for p in points3D.values()])
        errors = np.array([p.error for p in points3D.values()])
    else:
        points = np.zeros((0, 3))
        colors = np.zeros((0, 3), dtype=np.uint8)
        errors = np.array([])

    return cameras, images, points, colors, errors, cam_centers, cam_names

def project_point_to_image(point_3d, camera, image):
    """Project a 3D point to image coordinates."""
    R = qvec2rotmat(image.qvec)
    t = image.tvec

    # Transform to camera coords: P_cam = R @ P_world + t
    p_cam = R @ point_3d + t

    # Check if in front of camera
    if p_cam[2] <= 0:
        return None, None, False

    # Get intrinsics
    if camera.model == "PINHOLE":
        fx, fy, cx, cy = camera.params[:4]
    elif camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = camera.params[:3]
        fx = fy = f
    else:
        fx, fy = camera.params[0], camera.params[0]
        cx, cy = camera.width / 2, camera.height / 2

    # Project
    x = fx * p_cam[0] / p_cam[2] + cx
    y = fy * p_cam[1] / p_cam[2] + cy

    # Check bounds
    in_bounds = (0 <= x < camera.width) and (0 <= y < camera.height)

    return x, y, in_bounds

def analyze_point_cloud(colmap_path, frame_name="frame_000000"):
    """Analyze point cloud from a COLMAP reconstruction."""
    print(f"\n{'='*60}")
    print(f"Analyzing COLMAP at: {colmap_path}")
    print(f"{'='*60}")

    cameras, images, points, colors, errors, cam_centers, cam_names = load_colmap_points(colmap_path)

    print(f"\nCameras: {len(cameras)}")
    print(f"Images: {len(images)}")
    print(f"Points: {len(points)}")

    if len(points) == 0:
        print("No points found!")
        return None

    # Point cloud statistics
    print(f"\n--- Point Cloud Statistics ---")
    print(f"Points shape: {points.shape}")
    print(f"X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"Point cloud center: {points.mean(axis=0)}")
    print(f"Point cloud std: {points.std(axis=0)}")

    # Camera statistics
    print(f"\n--- Camera Statistics ---")
    print(f"Camera centers shape: {cam_centers.shape}")
    print(f"Cam X range: [{cam_centers[:, 0].min():.3f}, {cam_centers[:, 0].max():.3f}]")
    print(f"Cam Y range: [{cam_centers[:, 1].min():.3f}, {cam_centers[:, 1].max():.3f}]")
    print(f"Cam Z range: [{cam_centers[:, 2].min():.3f}, {cam_centers[:, 2].max():.3f}]")
    print(f"Camera center mean: {cam_centers.mean(axis=0)}")

    # Distance from cameras to points
    scene_center = cam_centers.mean(axis=0)
    point_dists = np.linalg.norm(points - scene_center, axis=1)
    cam_dists = np.linalg.norm(cam_centers - scene_center, axis=1)

    print(f"\n--- Distance Analysis ---")
    print(f"Scene center: {scene_center}")
    print(f"Camera distances from center: mean={cam_dists.mean():.3f}, max={cam_dists.max():.3f}")
    print(f"Point distances from center: mean={point_dists.mean():.3f}, max={point_dists.max():.3f}")
    print(f"Points within 2x cam radius: {(point_dists < 2*cam_dists.max()).sum()}/{len(points)}")
    print(f"Points within 5x cam radius: {(point_dists < 5*cam_dists.max()).sum()}/{len(points)}")

    # Error analysis
    if len(errors) > 0:
        print(f"\n--- Reprojection Error ---")
        print(f"Error range: [{errors.min():.3f}, {errors.max():.3f}]")
        print(f"Error mean: {errors.mean():.3f}")
        print(f"Error median: {np.median(errors):.3f}")
        print(f"Points with error < 2.0: {(errors < 2.0).sum()}/{len(errors)}")
        print(f"Points with error < 5.0: {(errors < 5.0).sum()}/{len(errors)}")

    # Test reprojection of a few points
    print(f"\n--- Reprojection Test (first 10 points) ---")
    test_images = list(images.values())[:3]  # Test with first 3 images

    n_test = min(10, len(points))
    for i in range(n_test):
        pt = points[i]
        print(f"\nPoint {i}: {pt}")
        for img in test_images:
            cam = cameras[img.camera_id]
            x, y, in_bounds = project_point_to_image(pt, cam, img)
            if x is not None:
                status = "IN" if in_bounds else "OUT"
                print(f"  -> {img.name[:20]}: ({x:.1f}, {y:.1f}) [{status}]")
            else:
                print(f"  -> {img.name[:20]}: BEHIND CAMERA")

    return {
        'points': points,
        'colors': colors,
        'errors': errors,
        'cam_centers': cam_centers,
        'cameras': cameras,
        'images': images
    }

def compare_reference_vs_roma(data_dir, frame_idx=0):
    """Compare reference COLMAP with RoMa-generated COLMAP."""
    data_path = Path(data_dir)

    # Reference COLMAP
    ref_path = data_path / "sparse" / "0"
    if ref_path.exists():
        print("\n" + "="*80)
        print("REFERENCE COLMAP (sparse/0)")
        print("="*80)
        ref_data = analyze_point_cloud(str(ref_path))
    else:
        print(f"Reference COLMAP not found at {ref_path}")
        ref_data = None

    # RoMa per-frame COLMAP
    roma_path = data_path / "sparse" / f"frame_{frame_idx:06d}"
    if roma_path.exists():
        print("\n" + "="*80)
        print(f"ROMA COLMAP (sparse/frame_{frame_idx:06d})")
        print("="*80)
        roma_data = analyze_point_cloud(str(roma_path))
    else:
        print(f"RoMa COLMAP not found at {roma_path}")
        roma_data = None

    # Compare coordinate systems
    if ref_data and roma_data:
        print("\n" + "="*80)
        print("COMPARISON: Reference vs RoMa")
        print("="*80)

        ref_cam_center = ref_data['cam_centers'].mean(axis=0)
        roma_cam_center = roma_data['cam_centers'].mean(axis=0)

        print(f"Reference camera center: {ref_cam_center}")
        print(f"RoMa camera center: {roma_cam_center}")
        print(f"Difference: {roma_cam_center - ref_cam_center}")

        # Check if cameras are the same
        ref_cam_scale = np.linalg.norm(ref_data['cam_centers'] - ref_cam_center, axis=1).max()
        roma_cam_scale = np.linalg.norm(roma_data['cam_centers'] - roma_cam_center, axis=1).max()

        print(f"\nReference scene scale (from cameras): {ref_cam_scale:.3f}")
        print(f"RoMa scene scale (from cameras): {roma_cam_scale:.3f}")
        print(f"Scale ratio: {roma_cam_scale / ref_cam_scale:.3f}")

        if len(ref_data['points']) > 0 and len(roma_data['points']) > 0:
            ref_pt_center = ref_data['points'].mean(axis=0)
            roma_pt_center = roma_data['points'].mean(axis=0)

            print(f"\nReference point cloud center: {ref_pt_center}")
            print(f"RoMa point cloud center: {roma_pt_center}")
            print(f"Difference: {roma_pt_center - ref_pt_center}")

    return ref_data, roma_data

def test_dataset_loading(data_dir):
    """Test loading through the FreeTime_dataset."""
    print("\n" + "="*80)
    print("TESTING DATASET LOADING")
    print("="*80)

    try:
        from datasets.FreeTime_dataset import FreeTimeParser, load_multiframe_colmap_points

        # Load parser to get transform
        parser = FreeTimeParser(
            data_dir=data_dir,
            factor=1,
            normalize=True,
            test_every=8
        )

        print(f"\nParser loaded successfully")
        print(f"Scene scale: {parser.scene_scale}")
        print(f"Scene center: {parser.scene_center}")
        print(f"Transform shape: {parser.transform.shape if hasattr(parser, 'transform') else 'N/A'}")

        # Check camera poses
        camtoworlds = parser.camtoworlds
        print(f"\nCamera poses shape: {camtoworlds.shape}")
        print(f"Camera X range: [{camtoworlds[:, 0, 3].min():.3f}, {camtoworlds[:, 0, 3].max():.3f}]")
        print(f"Camera Y range: [{camtoworlds[:, 1, 3].min():.3f}, {camtoworlds[:, 1, 3].max():.3f}]")
        print(f"Camera Z range: [{camtoworlds[:, 2, 3].min():.3f}, {camtoworlds[:, 2, 3].max():.3f}]")

        # Load multiframe points
        init_data = load_multiframe_colmap_points(
            data_dir,
            start_frame=0,
            end_frame=10,
            frame_step=10,
            max_error=2.0,
            match_threshold=0.1,
            transform=parser.transform
        )

        if init_data:
            positions, times, colors, velocities = init_data
            print(f"\nMultiframe points loaded:")
            print(f"  Positions shape: {positions.shape}")
            print(f"  Times shape: {times.shape}")
            print(f"  Colors shape: {colors.shape}")
            print(f"  Velocities shape: {velocities.shape}")

            print(f"\nPositions range:")
            print(f"  X: [{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]")
            print(f"  Y: [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}]")
            print(f"  Z: [{positions[:, 2].min():.3f}, {positions[:, 2].max():.3f}]")

            print(f"\nVelocities range:")
            print(f"  Vx: [{velocities[:, 0].min():.3f}, {velocities[:, 0].max():.3f}]")
            print(f"  Vy: [{velocities[:, 1].min():.3f}, {velocities[:, 1].max():.3f}]")
            print(f"  Vz: [{velocities[:, 2].min():.3f}, {velocities[:, 2].max():.3f}]")
            print(f"  Mean velocity magnitude: {np.linalg.norm(velocities, axis=1).mean():.3f}")

            # Check if points are within reasonable range of cameras
            cam_positions = camtoworlds[:, :3, 3]
            cam_center = cam_positions.mean(axis=0)
            cam_radius = np.linalg.norm(cam_positions - cam_center, axis=1).max()

            point_dists = np.linalg.norm(positions - cam_center, axis=1)
            print(f"\nCamera radius: {cam_radius:.3f}")
            print(f"Points within 2x camera radius: {(point_dists < 2*cam_radius).sum()}/{len(positions)}")
            print(f"Points within 5x camera radius: {(point_dists < 5*cam_radius).sum()}/{len(positions)}")
        else:
            print("No multiframe points loaded!")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--frame-idx", type=int, default=0, help="Frame index to analyze")
    args = parser.parse_args()

    # Compare reference vs RoMa COLMAP
    ref_data, roma_data = compare_reference_vs_roma(args.data_dir, args.frame_idx)

    # Test dataset loading
    test_dataset_loading(args.data_dir)
