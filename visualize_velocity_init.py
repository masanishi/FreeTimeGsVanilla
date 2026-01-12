#!/usr/bin/env python3
"""
Visualize velocity initialization to debug if velocities are correct.
Creates visualizations showing:
1. Point cloud at different times with motion applied
2. Velocity magnitude heatmap
3. Motion trajectories for sample points
"""

import sys
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

sys.path.insert(0, str(Path(__file__).parent))

from datasets.FreeTime_dataset import (
    FreeTimeParser,
    load_multiframe_colmap_grid_tracked,
)


def visualize_velocity_init(
    data_dir: str,
    output_dir: str = "velocity_debug",
    start_frame: int = 0,
    end_frame: int = 10,
    frame_step: int = 3,
    max_points: int = 50000,  # Subsample for visualization
):
    """Visualize velocity initialization."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    print("=" * 70)
    print("LOADING DATA WITH VELOCITY INITIALIZATION")
    print("=" * 70)

    # Load parser for transform
    parser = FreeTimeParser(
        data_dir=data_dir,
        factor=1,
        normalize=True,
        test_every=8
    )

    # Load with grid-tracked initialization
    init_data = load_multiframe_colmap_grid_tracked(
        data_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_step=frame_step,
        transform=parser.transform
    )

    positions = init_data['positions'].numpy()
    velocities = init_data['velocities'].numpy()
    times = init_data['times'].numpy().squeeze()
    colors = init_data['colors'].numpy()

    print(f"\nLoaded {len(positions)} points")
    print(f"Times range: [{times.min():.3f}, {times.max():.3f}]")
    print(f"Velocity magnitude: min={np.linalg.norm(velocities, axis=1).min():.6f}, "
          f"max={np.linalg.norm(velocities, axis=1).max():.6f}, "
          f"mean={np.linalg.norm(velocities, axis=1).mean():.6f}")

    # Subsample for visualization
    if len(positions) > max_points:
        idx = np.random.choice(len(positions), max_points, replace=False)
        positions = positions[idx]
        velocities = velocities[idx]
        times = times[idx]
        colors = colors[idx]
        print(f"Subsampled to {max_points} points for visualization")

    # =========================================================================
    # 1. Velocity Magnitude Histogram
    # =========================================================================
    print("\n[1] Creating velocity magnitude histogram...")
    vel_mag = np.linalg.norm(velocities, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of velocity magnitudes
    axes[0].hist(vel_mag, bins=100, color='blue', alpha=0.7)
    axes[0].set_xlabel('Velocity Magnitude')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Velocity Magnitude Distribution')
    axes[0].axvline(vel_mag.mean(), color='red', linestyle='--', label=f'Mean: {vel_mag.mean():.4f}')
    axes[0].legend()

    # Histogram of times
    axes[1].hist(times, bins=50, color='green', alpha=0.7)
    axes[1].set_xlabel('Time (mu_t)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Point Time Distribution')

    # Scatter: velocity vs time
    axes[2].scatter(times, vel_mag, alpha=0.1, s=1)
    axes[2].set_xlabel('Time (mu_t)')
    axes[2].set_ylabel('Velocity Magnitude')
    axes[2].set_title('Velocity vs Time')

    plt.tight_layout()
    plt.savefig(output_path / 'velocity_stats.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'velocity_stats.png'}")

    # =========================================================================
    # 2. Point Cloud at Different Times (applying motion)
    # =========================================================================
    print("\n[2] Creating point cloud visualizations at different times...")

    test_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig, axes = plt.subplots(1, len(test_times), figsize=(4*len(test_times), 4))

    for i, t in enumerate(test_times):
        # Apply motion: pos(t) = pos_init + vel * (t - mu_t)
        dt = t - times
        moved_positions = positions + velocities * dt[:, np.newaxis]

        # Project to 2D (top-down view: X-Z plane)
        ax = axes[i]
        scatter = ax.scatter(
            moved_positions[:, 0],
            moved_positions[:, 2],
            c=colors,  # Use actual colors
            s=0.5,
            alpha=0.5
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

    plt.suptitle('Point Cloud at Different Times (Top-Down View, X-Z)')
    plt.tight_layout()
    plt.savefig(output_path / 'pointcloud_over_time_xz.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'pointcloud_over_time_xz.png'}")

    # Side view (X-Y plane)
    fig, axes = plt.subplots(1, len(test_times), figsize=(4*len(test_times), 4))

    for i, t in enumerate(test_times):
        dt = t - times
        moved_positions = positions + velocities * dt[:, np.newaxis]

        ax = axes[i]
        scatter = ax.scatter(
            moved_positions[:, 0],
            moved_positions[:, 1],
            c=colors,
            s=0.5,
            alpha=0.5
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    plt.suptitle('Point Cloud at Different Times (Front View, X-Y)')
    plt.tight_layout()
    plt.savefig(output_path / 'pointcloud_over_time_xy.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'pointcloud_over_time_xy.png'}")

    # =========================================================================
    # 3. Velocity Direction Visualization (quiver plot)
    # =========================================================================
    print("\n[3] Creating velocity direction visualization...")

    # Subsample further for quiver plot
    quiver_idx = np.random.choice(len(positions), min(5000, len(positions)), replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Top-down (X-Z)
    ax = axes[0]
    ax.quiver(
        positions[quiver_idx, 0],
        positions[quiver_idx, 2],
        velocities[quiver_idx, 0],
        velocities[quiver_idx, 2],
        vel_mag[quiver_idx],
        cmap='hot',
        alpha=0.5,
        scale=1.0
    )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Velocity Directions (Top-Down, X-Z)')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    # Front view (X-Y)
    ax = axes[1]
    ax.quiver(
        positions[quiver_idx, 0],
        positions[quiver_idx, 1],
        velocities[quiver_idx, 0],
        velocities[quiver_idx, 1],
        vel_mag[quiver_idx],
        cmap='hot',
        alpha=0.5,
        scale=1.0
    )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Velocity Directions (Front View, X-Y)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(output_path / 'velocity_directions.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'velocity_directions.png'}")

    # =========================================================================
    # 4. Velocity Magnitude Heatmap (spatial)
    # =========================================================================
    print("\n[4] Creating velocity magnitude heatmap...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Top-down heatmap (X-Z)
    ax = axes[0]
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 2],
        c=vel_mag,
        cmap='hot',
        s=1,
        alpha=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Velocity Magnitude Heatmap (X-Z)')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    # Front view heatmap (X-Y)
    ax = axes[1]
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=vel_mag,
        cmap='hot',
        s=1,
        alpha=0.5
    )
    plt.colorbar(scatter, ax=ax, label='Velocity Magnitude')
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Velocity Magnitude Heatmap (X-Y)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.tight_layout()
    plt.savefig(output_path / 'velocity_heatmap.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'velocity_heatmap.png'}")

    # =========================================================================
    # 5. Motion Animation (GIF)
    # =========================================================================
    print("\n[5] Creating motion animation...")

    frames = []
    n_frames = 30

    for frame_i in range(n_frames):
        t = frame_i / (n_frames - 1)  # 0 to 1

        dt = t - times
        moved_positions = positions + velocities * dt[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(
            moved_positions[:, 0],
            moved_positions[:, 2],
            c=colors,
            s=0.5,
            alpha=0.5
        )
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title(f't = {t:.2f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Z')

        # Convert to image
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
        frames.append(img)
        plt.close()

    # Save as GIF
    import imageio
    gif_path = output_path / 'motion_animation.gif'
    imageio.mimsave(gif_path, frames, fps=10)
    print(f"  Saved: {gif_path}")

    # =========================================================================
    # 6. Sample Trajectories
    # =========================================================================
    print("\n[6] Visualizing sample point trajectories...")

    # Select points with highest velocity
    high_vel_idx = np.argsort(vel_mag)[-100:]  # Top 100 highest velocity

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot trajectories
    times_traj = np.linspace(0, 1, 20)

    for idx in high_vel_idx:
        trajectory = []
        for t in times_traj:
            dt = t - times[idx]
            pos = positions[idx] + velocities[idx] * dt
            trajectory.append(pos)
        trajectory = np.array(trajectory)

        # Color by time
        ax.plot(trajectory[:, 0], trajectory[:, 2], 'b-', alpha=0.3, linewidth=0.5)
        ax.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=10, marker='o')  # Start
        ax.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=10, marker='x')  # End

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title('Trajectories of Highest Velocity Points (Top-Down)')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')

    plt.tight_layout()
    plt.savefig(output_path / 'trajectories.png', dpi=150)
    plt.close()
    print(f"  Saved: {output_path / 'trajectories.png'}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VELOCITY INITIALIZATION SUMMARY")
    print("=" * 70)
    print(f"Total points: {len(positions)}")
    print(f"Points with zero velocity: {np.sum(vel_mag < 1e-6)}")
    print(f"Points with non-zero velocity: {np.sum(vel_mag >= 1e-6)}")
    print(f"Velocity magnitude:")
    print(f"  Min: {vel_mag.min():.6f}")
    print(f"  Max: {vel_mag.max():.6f}")
    print(f"  Mean: {vel_mag.mean():.6f}")
    print(f"  Std: {vel_mag.std():.6f}")
    print(f"\nMax displacement at t=1 from mu_t=0:")
    max_dt = 1.0
    max_disp = np.linalg.norm(velocities * max_dt, axis=1).max()
    print(f"  {max_disp:.4f} units")
    print(f"\nVisualization saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="velocity_debug")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=10)
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--max-points", type=int, default=50000)
    args = parser.parse_args()

    visualize_velocity_init(
        args.data_dir,
        args.output_dir,
        args.start_frame,
        args.end_frame,
        args.frame_step,
        args.max_points,
    )
