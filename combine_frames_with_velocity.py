#!/usr/bin/env python3
"""
Combine per-frame triangulated point clouds with velocity estimation.

This implements the Paper-Pure approach:
- Per-frame triangulation gives us 3D points at each time
- Velocity is estimated by k-NN matching between consecutive frames
- As per the paper: v = (P_{t+1} - P_t) / delta_t
- Since delta_t = 1 frame, velocity = displacement

Usage:
    python combine_frames_with_velocity.py \
        --input-dir /path/to/triangulation/output \
        --output-path /path/to/combined_with_velocity.npz \
        --frame-start 0 --frame-end 15 --frame-step 1
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import os
from scipy.spatial import cKDTree


def estimate_velocity_knn(
    points_t: np.ndarray,
    points_t1: np.ndarray,
    k: int = 1,
    max_distance: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate velocity using k-NN matching between consecutive frames.

    Paper's method: "3D points of two video frames are matched by k-nearest
    neighbor algorithm, and the translation between the point pairs are
    taken as the velocity of Gaussian primitives."

    Args:
        points_t: Points at time t, shape [N, 3]
        points_t1: Points at time t+1, shape [M, 3]
        k: Number of nearest neighbors (paper uses 1)
        max_distance: Maximum allowed distance for valid match

    Returns:
        velocities: Velocity vectors for points_t, shape [N, 3]
        valid_mask: Boolean mask for points with valid velocity, shape [N]
    """
    if len(points_t) == 0 or len(points_t1) == 0:
        return np.zeros_like(points_t), np.zeros(len(points_t), dtype=bool)

    # Build KD-tree for points at t+1
    tree = cKDTree(points_t1)

    # Find nearest neighbor for each point at time t
    distances, indices = tree.query(points_t, k=k)

    if k == 1:
        distances = distances.reshape(-1)
        indices = indices.reshape(-1)
    else:
        # Use closest match
        distances = distances[:, 0]
        indices = indices[:, 0]

    # Get matched points
    matched_points = points_t1[indices]

    # Compute displacement (velocity with dt=1)
    velocities = matched_points - points_t

    # Mark invalid matches (too far apart)
    valid_mask = distances < max_distance

    # Zero out invalid velocities
    velocities[~valid_mask] = 0.0

    return velocities, valid_mask


def combine_frames_with_velocity(
    input_dir: str,
    output_path: str,
    frame_start: int = 0,
    frame_end: int = 15,
    frame_step: int = 1,
    default_duration: float = 0.1,
    max_velocity_distance: float = 0.5,
    k_neighbors: int = 1,
    max_points_per_frame: Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Combine per-frame point clouds with velocity estimation.

    Args:
        input_dir: Directory containing points3d_frameXXXXXX.npy files
        output_path: Path to output NPZ file
        frame_start: First frame index
        frame_end: Last frame index (inclusive)
        frame_step: Step between frames
        default_duration: Temporal duration for each Gaussian
        max_velocity_distance: Max k-NN match distance for velocity
        k_neighbors: Number of neighbors for k-NN matching
        max_points_per_frame: Optional limit per frame
        verbose: Print progress

    Returns:
        Dictionary with combined data
    """
    input_dir = Path(input_dir)

    if verbose:
        print(f"\n{'='*60}")
        print("Combining Per-Frame Point Clouds with Velocity Estimation")
        print(f"{'='*60}")
        print(f"Input directory: {input_dir}")
        print(f"Frame range: {frame_start} to {frame_end} (step={frame_step})")
        print(f"Max velocity distance: {max_velocity_distance}")
        print(f"K-neighbors: {k_neighbors}")

    # Generate frame indices
    frame_indices = list(range(frame_start, frame_end + 1, frame_step))
    n_frames = len(frame_indices)

    if verbose:
        print(f"Total frames to process: {n_frames}")

    # First pass: Load all point clouds
    frame_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for frame_idx in frame_indices:
        points_path = input_dir / f"points3d_frame{frame_idx:06d}.npy"
        colors_path = input_dir / f"colors_frame{frame_idx:06d}.npy"

        if not points_path.exists():
            if verbose:
                print(f"  Warning: Missing {points_path}")
            continue

        positions = np.load(points_path).astype(np.float32)

        if colors_path.exists():
            colors = np.load(colors_path).astype(np.float32)
        else:
            colors = np.ones((len(positions), 3), dtype=np.float32) * 128

        # Optional subsampling
        if max_points_per_frame is not None and len(positions) > max_points_per_frame:
            indices = np.random.choice(len(positions), max_points_per_frame, replace=False)
            positions = positions[indices]
            colors = colors[indices]

        frame_data[frame_idx] = (positions, colors)

        if verbose:
            print(f"  Frame {frame_idx}: {len(positions):,} points")

    if len(frame_data) == 0:
        raise ValueError("No point clouds found!")

    # Second pass: Estimate velocities using k-NN matching
    if verbose:
        print(f"\n{'='*60}")
        print("Estimating velocities using k-NN matching...")
        print(f"{'='*60}")

    frame_velocities: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    total_valid = 0
    total_points = 0

    sorted_frames = sorted(frame_data.keys())

    for i, frame_idx in enumerate(sorted_frames):
        positions, _ = frame_data[frame_idx]
        n_points = len(positions)
        total_points += n_points

        # Check if there's a next frame
        if i < len(sorted_frames) - 1:
            next_frame_idx = sorted_frames[i + 1]
            next_positions, _ = frame_data[next_frame_idx]

            # Estimate velocity using k-NN
            velocities, valid_mask = estimate_velocity_knn(
                positions, next_positions,
                k=k_neighbors,
                max_distance=max_velocity_distance
            )

            n_valid = valid_mask.sum()
            total_valid += n_valid

            if verbose:
                vel_mag = np.linalg.norm(velocities[valid_mask], axis=1) if n_valid > 0 else np.array([0])
                print(f"  Frame {frame_idx} -> {next_frame_idx}: "
                      f"{n_valid:,}/{n_points:,} valid velocities "
                      f"(mean={vel_mag.mean():.4f}, max={vel_mag.max():.4f})")
        else:
            # Last frame: no velocity (or could use backward difference)
            velocities = np.zeros_like(positions)
            valid_mask = np.zeros(n_points, dtype=bool)
            if verbose:
                print(f"  Frame {frame_idx}: last frame, no velocity")

        frame_velocities[frame_idx] = (velocities, valid_mask)

    if verbose:
        print(f"\nTotal points with valid velocity: {total_valid:,}/{total_points:,} "
              f"({100*total_valid/total_points:.1f}%)")

    # Third pass: Combine all data
    if verbose:
        print(f"\n{'='*60}")
        print("Combining all frames...")
        print(f"{'='*60}")

    all_positions = []
    all_velocities = []
    all_colors = []
    all_times = []
    all_durations = []
    all_has_velocity = []

    time_range = frame_end - frame_start

    for frame_idx in sorted_frames:
        positions, colors = frame_data[frame_idx]
        velocities, valid_mask = frame_velocities[frame_idx]
        n_points = len(positions)

        # Compute normalized time
        t = (frame_idx - frame_start) / max(time_range, 1)

        times = np.full((n_points, 1), t, dtype=np.float32)
        durations = np.full((n_points, 1), default_duration, dtype=np.float32)

        all_positions.append(positions)
        all_velocities.append(velocities)
        all_colors.append(colors)
        all_times.append(times)
        all_durations.append(durations)
        all_has_velocity.append(valid_mask)

    # Concatenate
    positions = np.concatenate(all_positions, axis=0)
    velocities = np.concatenate(all_velocities, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    times = np.concatenate(all_times, axis=0)
    durations = np.concatenate(all_durations, axis=0)
    has_velocity = np.concatenate(all_has_velocity, axis=0)

    n_total = len(positions)

    if verbose:
        print(f"\nCombined Statistics:")
        print(f"  Total points: {n_total:,}")
        print(f"  Points with velocity: {has_velocity.sum():,} ({100*has_velocity.sum()/n_total:.1f}%)")
        print(f"  Time range: [{times.min():.3f}, {times.max():.3f}]")

        vel_mag = np.linalg.norm(velocities, axis=1)
        valid_vel = velocities[has_velocity]
        valid_mag = np.linalg.norm(valid_vel, axis=1) if len(valid_vel) > 0 else np.array([0])
        print(f"  Valid velocity magnitude: mean={valid_mag.mean():.6f}, max={valid_mag.max():.6f}")

    # Normalize colors to [0, 1]
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Save NPZ
    np.savez(
        output_path,
        # Main data
        positions=positions,
        velocities=velocities,
        colors=colors,
        times=times,
        durations=durations,
        has_velocity=has_velocity,
        # Metadata
        frame_start=frame_start,
        frame_end=frame_end,
        frame_step=frame_step,
        n_frames=n_frames,
        default_duration=default_duration,
        max_velocity_distance=max_velocity_distance,
        k_neighbors=k_neighbors,
        mode="per_frame_with_velocity",
    )

    if verbose:
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\nSaved to: {output_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"{'='*60}\n")

    return {
        'positions': positions,
        'velocities': velocities,
        'colors': colors,
        'times': times,
        'durations': durations,
        'has_velocity': has_velocity,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-frame point clouds with k-NN velocity estimation"
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing points3d_frameXXXXXX.npy files"
    )
    parser.add_argument(
        "--output-path", type=str, required=True,
        help="Output NPZ file path"
    )
    parser.add_argument(
        "--frame-start", type=int, default=0,
        help="First frame index"
    )
    parser.add_argument(
        "--frame-end", type=int, default=15,
        help="Last frame index (inclusive)"
    )
    parser.add_argument(
        "--frame-step", type=int, default=1,
        help="Step between frames"
    )
    parser.add_argument(
        "--default-duration", type=float, default=0.1,
        help="Default temporal duration (0.1 = visible for 10%% of sequence)"
    )
    parser.add_argument(
        "--max-velocity-distance", type=float, default=0.5,
        help="Max distance for k-NN velocity matching"
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=1,
        help="Number of nearest neighbors for velocity matching"
    )
    parser.add_argument(
        "--max-points-per-frame", type=int, default=None,
        help="Optional limit on points per frame"
    )

    args = parser.parse_args()

    combine_frames_with_velocity(
        input_dir=args.input_dir,
        output_path=args.output_path,
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        frame_step=args.frame_step,
        default_duration=args.default_duration,
        max_velocity_distance=args.max_velocity_distance,
        k_neighbors=args.k_neighbors,
        max_points_per_frame=args.max_points_per_frame,
    )


if __name__ == "__main__":
    main()
