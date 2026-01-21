#!/usr/bin/env python3
"""
Combine Keyframes with Velocity - Efficient version for keyframe-based training

This script:
1. Loads only KEYFRAME point clouds (not intermediate frames)
2. Computes velocity using the next frame (t → t+1)
3. Outputs only keyframe data with accurate velocity vectors

Why this is efficient:
- For 300 frames with keyframe_step=5:
  - Full combine: 300 frames × ~800k points = ~240M points
  - Keyframe combine: 60 keyframes × ~800k points = ~48M points (5x smaller!)
- Velocity is computed from consecutive pairs (t, t+1) for accuracy

Usage:
    python combine_keyframes_velocity.py \
        --input-dir /path/to/triangulation/output \
        --output-path /path/to/keyframes_with_velocity.npz \
        --frame-start 0 --frame-end 299 \
        --keyframe-step 5

Output NPZ contains:
- positions: [N, 3] - 3D positions from keyframes only
- velocities: [N, 3] - velocity in meters/frame (t → t+1)
- colors: [N, 3] - RGB colors
- times: [N, 1] - normalized time [0, 1]
- durations: [N, 1] - temporal duration (auto-computed based on keyframe_step)
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.spatial import cKDTree
from tqdm import tqdm
import os


def load_frame_data(input_dir: Path, frame_idx: int) -> tuple:
    """
    Load positions and colors for a single frame.

    Returns:
        positions: [N, 3] numpy array or None if file doesn't exist
        colors: [N, 3] numpy array or None
    """
    points_path = input_dir / f"points3d_frame{frame_idx:06d}.npy"
    colors_path = input_dir / f"colors_frame{frame_idx:06d}.npy"

    if not points_path.exists():
        return None, None

    positions = np.load(points_path).astype(np.float32)

    if colors_path.exists():
        colors = np.load(colors_path).astype(np.float32)
    else:
        colors = np.ones((len(positions), 3), dtype=np.float32) * 128

    return positions, colors


def compute_velocity_knn(
    pos_t: np.ndarray,
    pos_t1: np.ndarray,
    max_distance: float = 0.5,
    k: int = 1,
    n_workers: int = -1
) -> tuple:
    """
    Compute velocity for points at time t by finding nearest neighbors at t+1.

    Velocity = (P_{t+1} - P_t) / dt, where dt = 1 frame

    Args:
        pos_t: Positions at keyframe t [N, 3]
        pos_t1: Positions at frame t+1 [M, 3]
        max_distance: Maximum distance for valid velocity match
        k: Number of nearest neighbors
        n_workers: Number of workers for KDTree query (-1 = all cores)

    Returns:
        velocities: [N, 3] velocity vectors in meters/frame
        valid_mask: [N] boolean mask for points with valid velocity
    """
    if len(pos_t) == 0 or len(pos_t1) == 0:
        return np.zeros_like(pos_t), np.zeros(len(pos_t), dtype=bool)

    # Build KDTree for t+1 frame
    tree = cKDTree(pos_t1, balanced_tree=True, compact_nodes=True)

    # Find nearest neighbors
    distances, indices = tree.query(pos_t, k=k, workers=n_workers)

    if k > 1:
        distances = distances[:, 0]
        indices = indices[:, 0]

    # Compute velocity (displacement / dt, where dt=1 frame)
    velocities = np.zeros_like(pos_t)
    valid_mask = distances < max_distance

    if valid_mask.any():
        matched_positions = pos_t1[indices[valid_mask]]
        displacement = matched_positions - pos_t[valid_mask]
        velocities[valid_mask] = displacement  # dt = 1 frame, so velocity = displacement

    return velocities, valid_mask


def main():
    parser = argparse.ArgumentParser(
        description="Combine keyframes with velocity for efficient 4D training"
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
        "--frame-end", type=int, default=299,
        help="Last frame index"
    )
    parser.add_argument(
        "--keyframe-step", type=int, default=5,
        help="Step between keyframes (e.g., 5 means keyframes at 0, 5, 10, ...)"
    )
    parser.add_argument(
        "--max-velocity-distance", type=float, default=0.5,
        help="Maximum distance for k-NN velocity matching"
    )
    parser.add_argument(
        "--k-neighbors", type=int, default=1,
        help="Number of nearest neighbors for velocity matching"
    )
    parser.add_argument(
        "--sample-ratio", type=float, default=1.0,
        help="Sample ratio per keyframe (1.0 = keep all, 0.5 = keep 50%%)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate keyframe indices
    keyframes = list(range(args.frame_start, args.frame_end + 1, args.keyframe_step))
    n_keyframes = len(keyframes)
    total_frames = args.frame_end - args.frame_start + 1

    # Normalized time step between frames
    dt_normalized = 1.0 / total_frames  # For velocity scaling later

    # Duration that bridges the keyframe gap (3x the gap for smooth overlap)
    gap_normalized = args.keyframe_step / total_frames
    default_duration = gap_normalized * 3  # 3x overlap

    print("=" * 70)
    print("COMBINE KEYFRAMES WITH VELOCITY")
    print("=" * 70)
    print(f"Input directory: {input_dir}")
    print(f"Output path: {output_path}")
    print(f"\nFrame range: {args.frame_start} to {args.frame_end} ({total_frames} frames)")
    print(f"Keyframe step: {args.keyframe_step}")
    print(f"Number of keyframes: {n_keyframes}")
    print(f"Keyframes: {keyframes[:10]}..." if len(keyframes) > 10 else f"Keyframes: {keyframes}")
    print(f"\nVelocity Settings:")
    print(f"  Max distance: {args.max_velocity_distance}")
    print(f"  K-neighbors: {args.k_neighbors}")
    print(f"\nTemporal Settings:")
    print(f"  Keyframe gap (normalized): {gap_normalized:.4f}")
    print(f"  Auto duration (3x gap): {default_duration:.4f}")
    print(f"\nSampling:")
    print(f"  Sample ratio: {args.sample_ratio}")
    print("=" * 70)

    # Set random seed
    np.random.seed(args.seed)

    # Storage
    all_positions = []
    all_velocities = []
    all_colors = []
    all_times = []
    all_durations = []
    all_has_velocity = []

    # Stats
    total_points = 0
    total_valid_velocity = 0

    # Process each keyframe
    for i, keyframe in enumerate(tqdm(keyframes, desc="Processing keyframes")):
        next_frame = keyframe + 1

        # Load keyframe data
        positions, colors = load_frame_data(input_dir, keyframe)

        if positions is None:
            print(f"\n  Warning: Missing keyframe {keyframe}")
            continue

        n_points_original = len(positions)

        # Sample if requested
        if args.sample_ratio < 1.0:
            n_sample = int(len(positions) * args.sample_ratio)
            idx = np.random.choice(len(positions), n_sample, replace=False)
            positions = positions[idx]
            colors = colors[idx]

        n_points = len(positions)

        # Load next frame for velocity computation
        pos_next, _ = load_frame_data(input_dir, next_frame)

        if pos_next is not None and len(pos_next) > 0:
            # Compute velocity from t → t+1
            velocities, valid_mask = compute_velocity_knn(
                positions, pos_next,
                max_distance=args.max_velocity_distance,
                k=args.k_neighbors
            )
            n_valid = valid_mask.sum()
        else:
            # No next frame available (last keyframe or missing data)
            velocities = np.zeros_like(positions)
            valid_mask = np.zeros(n_points, dtype=bool)
            n_valid = 0

        # Compute normalized time for this keyframe
        t_normalized = (keyframe - args.frame_start) / total_frames
        times = np.full((n_points, 1), t_normalized, dtype=np.float32)
        durations = np.full((n_points, 1), default_duration, dtype=np.float32)

        # Store
        all_positions.append(positions)
        all_velocities.append(velocities)
        all_colors.append(colors)
        all_times.append(times)
        all_durations.append(durations)
        all_has_velocity.append(valid_mask)

        total_points += n_points
        total_valid_velocity += n_valid

    # Concatenate
    print("\nConcatenating...")
    positions = np.concatenate(all_positions, axis=0)
    velocities = np.concatenate(all_velocities, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    times = np.concatenate(all_times, axis=0)
    durations = np.concatenate(all_durations, axis=0)
    has_velocity = np.concatenate(all_has_velocity, axis=0)

    # Normalize colors to [0, 1] if needed
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Compute velocity statistics
    vel_mag = np.linalg.norm(velocities, axis=1)
    valid_vel_mag = vel_mag[has_velocity]

    print("\n" + "=" * 70)
    print("STATISTICS")
    print("=" * 70)
    print(f"Total keyframes processed: {len(all_positions)}/{n_keyframes}")
    print(f"Total points: {len(positions):,}")
    print(f"Points per keyframe: ~{len(positions) // len(all_positions):,}")
    print(f"\nVelocity Statistics:")
    print(f"  Points with valid velocity: {has_velocity.sum():,} ({100*has_velocity.sum()/len(positions):.1f}%)")
    if len(valid_vel_mag) > 0:
        print(f"  Valid velocity magnitude (m/frame):")
        print(f"    Mean: {valid_vel_mag.mean():.6f}")
        print(f"    Max:  {valid_vel_mag.max():.6f}")
        print(f"    Std:  {valid_vel_mag.std():.6f}")
    print(f"\nTime range: [{times.min():.4f}, {times.max():.4f}]")
    print(f"Unique time values: {len(np.unique(times))}")

    # Save NPZ
    print("\nSaving...")
    np.savez_compressed(
        output_path,
        # Main data
        positions=positions,
        velocities=velocities,  # In meters/frame (will be scaled by trainer)
        colors=colors,
        times=times,
        durations=durations,
        has_velocity=has_velocity,
        # Metadata
        frame_start=args.frame_start,
        frame_end=args.frame_end,
        keyframe_step=args.keyframe_step,
        n_keyframes=n_keyframes,
        max_velocity_distance=args.max_velocity_distance,
        k_neighbors=args.k_neighbors,
        sample_ratio=args.sample_ratio,
        mode="keyframes_with_velocity",
    )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
    print(f"Saved to: {output_path}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"\nNext step: Train with paper_efficient config:")
    print(f"  python simple_trainer_freetime_4d_pure_relocation.py paper_efficient \\")
    print(f"      --init-npz-path {output_path} \\")
    print(f"      --start-frame 0 --end-frame {total_frames}")
    print("=" * 70)


if __name__ == "__main__":
    main()
