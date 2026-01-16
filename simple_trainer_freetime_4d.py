"""
FreeTimeGS: 4D Gaussian Splatting Implementation
================================================

A complete implementation of 4D Gaussian Splatting for dynamic scene reconstruction,
based on the FreeTimeGS paper methodology.

Core Methodology
----------------
Each Gaussian has 8 learnable parameter groups:
    1. Position (µx): [N, 3] - Canonical 3D position
    2. Time (µt): [N, 1] - Canonical time (when Gaussian is most visible)
    3. Duration (s): [N, 1] - Temporal width (how long Gaussian is visible)
    4. Velocity (v): [N, 3] - Linear velocity vector
    5. Scale: [N, 3] - 3D scale (log space)
    6. Quaternion: [N, 4] - Rotation orientation
    7. Opacity (σ): [N] - Base opacity (logit space)
    8. Spherical Harmonics: [N, K, 3] - View-dependent color

Key Equations:
    - Motion: µx(t) = µx + v·(t - µt)
    - Temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)²)
    - Combined opacity: σ_final = σ · σ(t)
    - 4D Regularization: Lreg(t) = (1/N) * Σ(σ · stop_gradient[σ(t)])

Training Phases
---------------
1. WARMUP (steps 0 to warmup_steps):
   - Basic appearance learning (colors, scales, opacity)
   - No temporal opacity (all Gaussians visible)
   - All temporal params frozen (times, durations, velocities)

2. CANONICAL (steps warmup_steps to warmup_steps + canonical_phase_steps):
   - Static positions (velocities not applied)
   - Temporal opacity active (learn visibility width)
   - Only durations optimized (times FROZEN to preserve velocity relationship)

3. FULL 4D (steps canonical_phase_steps+):
   - Motion enabled: µx(t) = µx + v·(t-µt)
   - All temporal params optimized (times, durations, velocities)
   - 4D regularization active
   - Periodic relocation of low-opacity Gaussians

Outputs
-------
At the end of training, the following are generated:

1. Checkpoints: {result_dir}/ckpts/ckpt_{step}.pt
   - Contains: splats state_dict, optimizer states, step number
   - Can be used to resume training or export

2. Trajectory Videos: {result_dir}/videos/
   - traj_4d_step{step}.mp4: RGB render with smooth camera + time progression
   - traj_duration_step{step}.mp4: Duration heatmap visualization
   - traj_velocity_step{step}.mp4: Velocity magnitude heatmap visualization

3. PLY Sequence: {result_dir}/ply_sequence_step{step}/
   - frame_000000.ply to frame_XXXXXX.ply
   - One PLY per frame with positions/opacities computed for that time

4. TensorBoard Logs: {result_dir}/tb/
   - Loss curves, metrics, histograms
   - Duration/velocity visualizations every 500 steps

Usage Examples
--------------
# Basic training with MCMC strategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \\
    --data-dir /path/to/data \\
    --windowed-npz-path /path/to/windowed_points.npz \\
    --result-dir /path/to/results \\
    --start-frame 0 --end-frame 300

# Training with DefaultStrategy
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py default \\
    --data-dir /path/to/data \\
    --windowed-npz-path /path/to/windowed_points.npz \\
    --result-dir /path/to/results

# Resume training from checkpoint
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \\
    --data-dir /path/to/data \\
    --windowed-npz-path /path/to/windowed_points.npz \\
    --result-dir /path/to/results \\
    --ckpt-path /path/to/results/ckpts/ckpt_29999.pt

# Export PLY sequence and videos from checkpoint (no training)
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \\
    --data-dir /path/to/data \\
    --windowed-npz-path /path/to/windowed_points.npz \\
    --result-dir /path/to/results \\
    --ckpt-path /path/to/results/ckpts/ckpt_59999.pt \\
    --export-only

# Custom training parameters
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \\
    --data-dir /path/to/data \\
    --windowed-npz-path /path/to/windowed_points.npz \\
    --result-dir /path/to/results \\
    --max-steps 60000 \\
    --velocity-lr-start 1e-2 \\
    --velocity-lr-end 5e-4 \\
    --init-duration 0.05 \\
    --render-traj-n-frames 240 \\
    --export-ply-format ply

Key Configuration Options
-------------------------
Data:
    --data-dir: Path to dataset with images and COLMAP sparse reconstruction
    --windowed-npz-path: Path to windowed_points.npz from triangulation
    --start-frame, --end-frame: Frame range to use

Training:
    --max-steps: Total training iterations (default: 60000)
    --warmup-steps: Warmup phase length (default: 500)
    --canonical-phase-steps: Canonical phase length (default: 2000)

Temporal:
    --init-duration: Initial duration for Gaussians (default: 0.1)
    --velocity-lr-start: Starting velocity learning rate (default: 5e-3)
    --velocity-lr-end: Ending velocity learning rate (default: 1e-4)

Regularization:
    --lambda-4d-reg: 4D regularization weight (default: 1e-3)
    --lambda-duration-reg: Duration regularization weight (default: 5e-4)

Checkpoint & Export:
    --ckpt-path: Path to checkpoint for resume/export
    --export-only: If set, only export PLY/videos from checkpoint
    --export-ply: Enable PLY sequence export (default: True)
    --export-ply-format: PLY format - "ply", "splat", or "ply_compressed"

Visualization:
    --render-traj-path: Trajectory type - "interp" or "ellipse"
    --render-traj-n-frames: Number of frames in trajectory video (default: 120)
    --render-traj-fps: Video FPS (default: 30)
"""

import json
import math
import os
import time
import shutil
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import scipy.interpolate
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.exporter import export_splats

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.FreeTime_dataset import FreeTimeParser, FreeTimeDataset
from utils import knn, rgb_to_sh, set_random_seed


# ============================================================
# Trajectory Generation Utilities (adapted from multinerf)
# ============================================================

def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def _viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix."""
    vec2 = _normalize(lookdir)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


def _focus_point_fn(poses: np.ndarray) -> np.ndarray:
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.05,
    rot_weight: float = 0.1,
) -> np.ndarray:
    """Creates a smooth spline path between input keyframe camera poses.

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        n_interp: returned path will have n_interp * (n - 1) total poses.
        spline_degree: polynomial degree of B-spline.
        smoothness: parameter for spline smoothing, 0 forces exact interpolation.
        rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
        Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """
    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([_viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(
        points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness
    )
    return points_to_poses(new_points)


def generate_ellipse_path_z(
    poses: np.ndarray,
    n_frames: int = 120,
    variation: float = 0.0,
    phase: float = 0.0,
    height: float = 0.0,
) -> np.ndarray:
    """Generate an elliptical render path based on the given poses."""
    center = _focus_point_fn(poses)
    offset = np.array([center[0], center[1], height])

    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)
    low = -sc + offset
    high = sc + offset
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        return np.stack(
            [
                low[0] + (high - low)[0] * (np.cos(theta) * 0.5 + 0.5),
                low[1] + (high - low)[1] * (np.sin(theta) * 0.5 + 0.5),
                variation * (
                    z_low[2] + (z_high - z_low)[2] * (np.cos(theta + 2 * np.pi * phase) * 0.5 + 0.5)
                ) + height,
            ],
            -1,
        )

    theta = np.linspace(0, 2.0 * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta)
    positions = positions[:-1]

    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])

    return np.stack([_viewmatrix(center - p, up, p) for p in positions])


@dataclass
class Config:
    """
    Configuration for FreeTimeGS 4D Gaussian Splatting Training.

    This configuration controls all aspects of training including data loading,
    optimization, regularization, checkpointing, and export options.
    """

    # ==================== Data Paths ====================
    data_dir: str = "data/4d_scene"
    """Path to dataset directory containing images and COLMAP sparse reconstruction."""

    result_dir: str = "results/freetime_4d"
    """Output directory for checkpoints, videos, PLY files, and tensorboard logs."""

    windowed_npz_path: Optional[str] = None
    """Path to windowed_points.npz from triangulation. Contains initial positions,
    velocities, colors, times, and durations for Gaussians."""

    data_factor: int = 1
    """Downsample factor for images. 1 = full resolution, 2 = half, etc."""

    test_every: int = 8
    """Use every N-th camera for validation (others used for training)."""

    # ==================== Frame Range ====================
    start_frame: int = 0
    """Starting frame index (inclusive). Time t=0 corresponds to this frame."""

    end_frame: int = 300
    """Ending frame index (exclusive). Time t=1 corresponds to this frame."""
    frame_step: int = 1
    """Step between frames when loading data."""

    # ==================== Sampling from Windowed NPZ ====================
    max_samples: int = 2_000_000
    """Maximum number of Gaussians to initialize from NPZ file."""

    sample_n_times: int = 10
    """Number of time points to sample from for better temporal coverage."""

    sample_high_velocity_ratio: float = 0.8
    """Ratio of high-velocity points to prioritize during sampling (0.0-1.0)."""

    # ==================== Training ====================
    max_steps: int = 70_000
    """Total number of training iterations."""

    batch_size: int = 1
    """Batch size for training (number of images per iteration)."""

    steps_scaler: float = 1.0
    """Scale factor for all step-related parameters (for quick experiments)."""

    eval_steps: List[int] = field(default_factory=lambda: [15_000, 30_000, 45_000, 60_000])
    """Steps at which to run evaluation on validation set."""

    save_steps: List[int] = field(default_factory=lambda: [15_000, 30_000, 45_000, 60_000])
    """Steps at which to save checkpoints."""

    eval_sample_every: int = 60
    """Evaluate every N-th frame for faster validation (e.g., 300/60 = 5 frames)."""

    # ==================== Model ====================
    sh_degree: int = 3
    """Maximum spherical harmonics degree for view-dependent color."""

    sh_degree_interval: int = 1000
    """Steps between increasing SH degree (starts at 0, increases to sh_degree)."""

    init_opacity: float = 0.5
    """Initial opacity for Gaussians (before sigmoid)."""

    init_scale: float = 1.0
    """Scale factor for initial Gaussian sizes (computed from KNN distances)."""

    init_duration: float = 0.1
    """Initial temporal duration for Gaussians. Smaller = sharper temporal profiles.
    A duration of 0.1 means Gaussian is visible for ~10% of the sequence."""

    # ==================== Loss Weights ====================
    lambda_img: float = 0.8
    """Weight for L1 image reconstruction loss."""

    lambda_ssim: float = 0.2
    """Weight for SSIM structural similarity loss."""

    lambda_perc: float = 0.01
    """Weight for LPIPS perceptual loss."""

    lambda_4d_reg: float = 1e-3
    """Weight for 4D regularization loss: Lreg = (1/N) * Σ(σ * stop_grad[σ(t)]).
    Encourages Gaussians to have high opacity at their canonical time."""

    lambda_duration_reg: float = 5e-4
    """Weight for duration regularization. Penalizes wide temporal windows
    that cause temporal blur. Only applied to durations > init_duration."""

    # ==================== Training Phases ====================
    warmup_steps: int = 500
    """Duration of warmup phase. During warmup:
    - Basic appearance learning (colors, scales, opacity)
    - No temporal opacity (all Gaussians visible)
    - All temporal params frozen (times, durations, velocities)"""

    canonical_phase_steps: int = 2000
    """Duration of canonical phase (after warmup). During canonical:
    - Static positions (velocities not applied)
    - Temporal opacity active (learn visibility width)
    - Only durations optimized (times FROZEN to preserve velocity relationship)"""

    # ==================== Learning Rates ====================
    position_lr: float = 1.6e-4
    """Learning rate for Gaussian positions."""

    scales_lr: float = 5e-3
    """Learning rate for Gaussian scales."""

    quats_lr: float = 1e-3
    """Learning rate for Gaussian rotations (quaternions)."""

    opacities_lr: float = 5e-2
    """Learning rate for Gaussian opacities."""

    sh0_lr: float = 2.5e-3
    """Learning rate for DC spherical harmonics (base color)."""

    shN_lr: float = 2.5e-3 / 20
    """Learning rate for higher-order spherical harmonics."""

    times_lr: float = 5e-4
    """Learning rate for canonical times."""

    durations_lr: float = 1e-3
    """Learning rate for temporal durations."""

    velocity_lr_start: float = 1e-3
    """Starting learning rate for velocities (annealed during training)."""

    velocity_lr_end: float = 1e-5
    """Ending learning rate for velocities."""

    # ==================== Periodic Relocation ====================
    use_relocation: bool = True
    """Enable periodic relocation of low-opacity Gaussians to high-gradient regions."""

    relocation_every: int = 100
    """Relocate Gaussians every N iterations."""

    relocation_start_iter: int = 500
    """Start relocation after this many iterations."""

    relocation_stop_iter: int = 50_000
    """Stop relocation after this many iterations (allow fine-tuning)."""

    relocation_opacity_threshold: float = 0.005
    """Gaussians with opacity below this are considered 'dead' and may be relocated."""

    relocation_max_ratio: float = 0.015
    """Maximum fraction of Gaussians to relocate per step (1.5% default, prevents scene darkening)."""

    relocation_lambda_grad: float = 0.5
    """Weight for gradient magnitude in relocation sampling score."""

    relocation_lambda_opa: float = 0.5
    """Weight for opacity in relocation sampling score."""

    # ==================== Pruning (Disabled by Default) ====================
    use_pruning: bool = False
    """Enable custom pruning. Disabled by default as DefaultStrategy handles pruning."""

    prune_every: int = 500
    """Prune Gaussians every N iterations (if enabled)."""

    prune_start_iter: int = 2000
    """Start pruning after this many iterations."""

    prune_stop_iter: int = 20_000
    """Stop pruning after this many iterations."""

    prune_opacity_threshold: float = 0.005
    """Prune Gaussians with opacity below this threshold."""

    # ==================== Motion-Aware Loss ====================
    use_motion_weighted_loss: bool = True
    """Enable difficulty-weighted photometric loss. Regions with high velocity
    and low duration (fast-moving, short-lived) get higher loss weight."""

    motion_weight_velocity: float = 0.45
    """Weight for velocity magnitude in difficulty score (0-1).
    Higher = more emphasis on high-velocity regions."""

    motion_weight_duration: float = 0.55
    """Weight for inverse duration in difficulty score (0-1).
    Higher = more emphasis on short-duration (transient) regions."""

    motion_weight_max: float = 3.0
    """Maximum difficulty weight multiplier. Prevents extreme weights
    that could destabilize training. Range: [1, motion_weight_max]."""

    motion_weight_start_iter: int = 15000
    """Start applying motion-weighted loss after this iteration.
    Should be well after canonical phase so velocities/durations are established first."""

    # ==================== Temporal Coverage Densification ====================
    use_temporal_densification: bool = True
    """Enable temporal coverage densification. Spawns new Gaussians in regions
    where temporal coverage is poor (high velocity + low duration)."""

    temporal_densify_every: int = 500
    """Check for temporal coverage gaps every N iterations."""

    temporal_densify_start_iter: int = 15000
    """Start temporal densification after this iteration.
    Should wait until velocities/durations are learned to identify coverage gaps."""

    temporal_densify_stop_iter: int = 40000
    """Stop temporal densification after this iteration."""

    temporal_densify_velocity_threshold: float = 0.1
    """Velocity magnitude threshold for identifying fast-moving Gaussians.
    Gaussians with velocity > threshold are candidates for temporal spawning."""

    temporal_densify_duration_threshold: float = 0.15
    """Duration threshold for identifying short-lived Gaussians.
    Gaussians with duration < threshold are candidates for temporal spawning."""

    temporal_densify_max_spawn: int = 5000
    """Maximum number of new Gaussians to spawn per densification step."""

    temporal_densify_opacity_threshold: float = 0.3
    """Only consider Gaussians with opacity above this for temporal spawning.
    Prevents spawning from already-faint Gaussians."""

    # ==================== Rendering ====================
    near_plane: float = 0.01
    """Near clipping plane for rendering."""

    far_plane: float = 1e10
    """Far clipping plane for rendering."""

    packed: bool = False
    """Use packed mode for rasterization (more memory efficient for large scenes)."""

    antialiased: bool = False
    """Use antialiased rasterization mode."""

    # ==================== Trajectory Rendering ====================
    render_traj_path: Literal["interp", "ellipse"] = "interp"
    """Trajectory type for video rendering:
    - 'interp': Smooth spline interpolation through camera poses
    - 'ellipse': Elliptical path around the scene center"""

    render_traj_n_frames: int = 120
    """Number of frames in the trajectory video."""

    render_traj_fps: int = 30
    """Frames per second for trajectory video."""

    # ==================== PLY Export ====================
    export_ply: bool = True
    """Export PLY sequence at end of training. One PLY per frame with
    positions and opacities computed for that specific time point."""

    export_ply_format: Literal["ply", "splat", "ply_compressed"] = "ply"
    """PLY export format:
    - 'ply': Standard PLY format (supported by most viewers)
    - 'splat': Custom format for antimatter15 viewer
    - 'ply_compressed': Compressed format for Supersplat viewer"""

    export_ply_opacity_threshold: float = 0.01
    """Only export Gaussians with combined opacity (base × temporal) above this threshold.
    Higher = smaller files but may miss faint Gaussians. 0.01 = 1% opacity cutoff."""

    # ==================== Strategy ====================
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=lambda: DefaultStrategy(verbose=True)
    )
    """Densification strategy: DefaultStrategy or MCMCStrategy.
    Controls how Gaussians are split, cloned, and pruned during training."""

    # ==================== Miscellaneous ====================
    global_scale: float = 1.0
    """Global scale factor for the scene."""

    tb_every: int = 100
    """Log scalar metrics to TensorBoard every N steps (loss, PSNR, etc.)."""

    tb_image_every: int = 200
    """Log images to TensorBoard every N steps (ground truth vs rendered)."""

    disable_viewer: bool = True
    """Disable the interactive Viser viewer. Set to False to enable 3D viewer at localhost:8080."""

    lpips_net: Literal["vgg", "alex"] = "alex"
    """Network architecture for LPIPS perceptual loss:
    - 'alex': AlexNet-based (faster, default)
    - 'vgg': VGG-based (slightly more accurate)"""

    # ==================== Checkpoint & Resume ====================
    ckpt_path: Optional[str] = None
    """Path to checkpoint file (.pt) to resume training from or export from.
    Checkpoint contains: splats state_dict, optimizer states, and step number."""

    export_only: bool = False
    """If True and ckpt_path is provided, load checkpoint and export PLY/videos
    without training. Useful for generating outputs from a trained model."""

    def adjust_steps(self, factor: float):
        """Scale training steps by factor."""
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.warmup_steps = int(self.warmup_steps * factor)
        self.canonical_phase_steps = int(self.canonical_phase_steps * factor)
        self.relocation_start_iter = int(self.relocation_start_iter * factor)
        self.relocation_stop_iter = int(self.relocation_stop_iter * factor)
        self.prune_start_iter = int(self.prune_start_iter * factor)
        self.prune_stop_iter = int(self.prune_stop_iter * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)


def load_windowed_npz(
    npz_path: str,
    max_samples: int = 2_000_000,
    n_times: int = 3,
    high_velocity_ratio: float = 0.8,
    frame_start: int = 0,
    frame_end: int = 300,
    transform: Optional[np.ndarray] = None,
) -> Dict[str, torch.Tensor]:
    """
    Load and sample from windowed NPZ file.

    NPZ format (from triangulate_windowed.py):
    - positions: [N, 3] - 3D positions
    - velocities: [N, 3] - linear velocities (vx, vy, vz)
    - colors: [N, 3] - RGB colors
    - times: [N, 1] - normalized time in [0, 1]
    - durations: [N, 1] - temporal window widths
    """
    print(f"\n[WindowedNPZ] Loading: {npz_path}")

    data = np.load(npz_path)
    positions = data['positions'].astype(np.float32)
    velocities = data['velocities'].astype(np.float32)  # [N, 3] - velocity in x, y, z
    colors = data['colors'].astype(np.float32)
    times = data['times'].flatten().astype(np.float32)
    durations = data['durations'].flatten().astype(np.float32) if 'durations' in data else np.ones_like(times) * 0.1

    n_total = len(positions)
    print(f"  Total points: {n_total:,}")
    print(f"  Velocity shape: {velocities.shape} (x, y, z)")
    print(f"  Time range: [{times.min():.3f}, {times.max():.3f}]")

    # Normalize colors
    if colors.max() > 1.0:
        colors = colors / 255.0

    # Sample if needed
    if max_samples > 0 and n_total > max_samples:
        print(f"\n  [Sampling] Reducing {n_total:,} to {max_samples:,}")

        # Get unique times
        unique_times = np.unique(times)
        n_available = len(unique_times)
        actual_n_times = min(n_times, n_available)

        # Select time windows evenly
        if actual_n_times == n_available:
            selected_times = unique_times
        else:
            indices = np.linspace(0, len(unique_times)-1, actual_n_times, dtype=int)
            selected_times = unique_times[indices]

        print(f"    Sampling from {actual_n_times} times: {selected_times.round(3)}")

        # Compute velocity magnitudes
        vel_mag = np.linalg.norm(velocities, axis=1)

        # Sample from each time
        samples_per_time = max_samples // actual_n_times
        high_vel_per_time = int(samples_per_time * high_velocity_ratio)
        spatial_per_time = samples_per_time - high_vel_per_time

        all_indices = []
        for t in selected_times:
            t_mask = np.abs(times - t) < 0.01
            t_indices = np.where(t_mask)[0]
            n_at_time = len(t_indices)

            if n_at_time == 0:
                continue

            # High velocity sampling
            t_vel = vel_mag[t_indices]
            n_high = min(high_vel_per_time, n_at_time)
            if n_high > 0:
                sorted_idx = np.argsort(t_vel)[::-1]
                high_indices = t_indices[sorted_idx[:n_high]]
                all_indices.extend(high_indices.tolist())

            # Random spatial sampling for rest
            n_spatial = min(spatial_per_time, n_at_time)
            if n_spatial > 0:
                remaining = np.setdiff1d(t_indices, high_indices if n_high > 0 else np.array([]))
                if len(remaining) > 0:
                    spatial_sample = np.random.choice(remaining, min(n_spatial, len(remaining)), replace=False)
                    all_indices.extend(spatial_sample.tolist())

        # Remove duplicates and shuffle
        all_indices = list(set(all_indices))
        np.random.shuffle(all_indices)
        all_indices = np.array(all_indices[:max_samples])

        positions = positions[all_indices]
        velocities = velocities[all_indices]
        colors = colors[all_indices]
        times = times[all_indices]
        durations = durations[all_indices]

        print(f"    Sampled to {len(positions):,} points")

    # Apply transform if provided
    if transform is not None:
        R = transform[:3, :3]
        t = transform[:3, 3]
        positions = (positions @ R.T) + t
        # Velocities are direction vectors - only rotate, no translate
        velocities = velocities @ R.T

    # Cap velocities to reasonable range
    vel_mag = np.linalg.norm(velocities, axis=1, keepdims=True)
    max_vel = 0.5
    large = vel_mag.squeeze() > max_vel
    if large.any():
        scale = np.clip(max_vel / (vel_mag + 1e-8), a_min=None, a_max=1.0)
        velocities = velocities * scale
        print(f"  Capped {large.sum()} velocities to {max_vel}")

    print(f"\n[WindowedNPZ] Final: {len(positions):,} points")
    print(f"  Time range: [{times.min():.3f}, {times.max():.3f}]")
    vel_mags = np.linalg.norm(velocities, axis=1)
    print(f"  Velocity [vx, vy, vz] magnitude: [{vel_mags.min():.6f}, {vel_mags.max():.6f}]")

    return {
        'positions': torch.from_numpy(positions),
        'velocities': torch.from_numpy(velocities),  # [N, 3] - vx, vy, vz
        'colors': torch.from_numpy(colors),
        'times': torch.from_numpy(times).unsqueeze(-1),
        'durations': torch.from_numpy(durations).unsqueeze(-1),
    }


def create_splats_with_optimizers_4d(
    cfg: Config,
    init_data: Dict[str, torch.Tensor],
    scene_scale: float = 1.0,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Create 4D Gaussian splats with temporal parameters.

    Parameters per Gaussian (8 from paper + extras):
    - means: [N, 3] - canonical position µx
    - scales: [N, 3] - log scales
    - quats: [N, 4] - quaternion orientation
    - opacities: [N] - logit of base opacity σ
    - sh0: [N, 1, 3] - DC spherical harmonics
    - shN: [N, K, 3] - higher-order SH (K = (sh_degree+1)^2 - 1)
    - times: [N, 1] - canonical time µt
    - durations: [N, 1] - log of temporal duration s
    - velocities: [N, 3] - linear velocity v
    """
    points = init_data['positions']
    velocities = init_data['velocities']
    colors = init_data['colors']
    times = init_data['times']
    durations = init_data['durations']

    N = len(points)

    # Compute scales from KNN
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg).clamp(min=1e-6)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

    # Initialize parameters
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), cfg.init_opacity))

    # Durations: Use larger default to ensure temporal coverage
    # NPZ durations are often too small (window_size/total_frames), causing black frames
    # A duration of 0.2 means each Gaussian is visible for ~20% of the sequence
    # which provides good overlap between time samples
    min_duration = cfg.init_duration  # Default 0.2
    if durations.min() > 0:
        # Use max of NPZ duration and min_duration to ensure coverage
        durations_clamped = torch.clamp(durations, min=min_duration)
        log_durations = torch.log(durations_clamped)
        print(f"[Init] NPZ durations: [{durations.min():.3f}, {durations.max():.3f}]")
        print(f"[Init] Using clamped durations: [{durations_clamped.min():.3f}, {durations_clamped.max():.3f}]")
    else:
        log_durations = torch.log(torch.full((N, 1), min_duration))
        print(f"[Init] Using config init_duration: {min_duration}")

    # SH colors
    sh_colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    sh_colors[:, 0, :] = rgb_to_sh(colors)

    # Create parameter dict
    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), cfg.position_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
        ("sh0", torch.nn.Parameter(sh_colors[:, :1, :]), cfg.sh0_lr),
        ("shN", torch.nn.Parameter(sh_colors[:, 1:, :]), cfg.shN_lr),
        # Temporal parameters
        ("times", torch.nn.Parameter(times), cfg.times_lr),
        ("durations", torch.nn.Parameter(log_durations), cfg.durations_lr),
        ("velocities", torch.nn.Parameter(velocities), cfg.velocity_lr_start),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Create optimizers with batch size scaling
    BS = cfg.batch_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers


class FreeTime4DRunner:
    """FreeTimeGS 4D Gaussian Splatting Trainer."""

    def __init__(self, local_rank: int, world_rank: int, world_size: int, cfg: Config):
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Setup directories
        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load dataset
        self.parser = FreeTimeParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
        )
        # Create test_set: every N-th camera for validation
        num_cameras = len(self.parser.camera_names)
        test_set = list(range(0, num_cameras, cfg.test_every))
        print(f"[FreeTime4D] Using {len(test_set)} cameras for validation (every {cfg.test_every}-th of {num_cameras})")

        self.trainset = FreeTimeDataset(self.parser, split="train", test_set=test_set)
        self.valset = FreeTimeDataset(self.parser, split="val", test_set=test_set)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"[FreeTime4D] Scene scale: {self.scene_scale}")
        print(f"[FreeTime4D] Train: {len(self.trainset)}, Val: {len(self.valset)}")

        # Load windowed NPZ and initialize Gaussians
        if cfg.windowed_npz_path is None or not os.path.exists(cfg.windowed_npz_path):
            raise ValueError(f"Windowed NPZ not found: {cfg.windowed_npz_path}")

        transform = self.parser.transform if hasattr(self.parser, 'transform') else None
        init_data = load_windowed_npz(
            cfg.windowed_npz_path,
            max_samples=cfg.max_samples,
            n_times=cfg.sample_n_times,
            high_velocity_ratio=cfg.sample_high_velocity_ratio,
            frame_start=cfg.start_frame,
            frame_end=cfg.end_frame,
            transform=transform,
        )

        # Filter distant points
        points = init_data['positions']
        max_dist = 5.0 * self.scene_scale
        dists = torch.norm(points, dim=1)
        valid = dists < max_dist

        for key in init_data:
            init_data[key] = init_data[key][valid]

        print(f"[FreeTime4D] After filtering: {len(init_data['positions']):,} Gaussians")

        # Create splats and optimizers
        self.splats, self.optimizers = create_splats_with_optimizers_4d(
            cfg, init_data, self.scene_scale, self.device
        )
        print(f"[FreeTime4D] Initialized {len(self.splats['means']):,} Gaussians")

        # Strategy state (for DefaultStrategy or MCMCStrategy)
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Gradient accumulator for relocation sampling score
        self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)
        self.grad_count = 0

        # Losses
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        else:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(self.device)

        # Track starting step for resume
        self.start_step = 0

        # Load checkpoint if provided
        if cfg.ckpt_path is not None:
            self.load_checkpoint(cfg.ckpt_path)

    def load_checkpoint(self, ckpt_path: str):
        """Load model and optimizer states from checkpoint."""
        print(f"\n[Checkpoint] Loading from: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        # Reinitialize splats from checkpoint (handles size mismatch from densification)
        ckpt_splats = ckpt["splats"]
        N = ckpt_splats["means"].shape[0]
        print(f"  Checkpoint has {N:,} Gaussians")

        # Replace splats with checkpoint data
        self.splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(ckpt_splats["means"]),
            "scales": torch.nn.Parameter(ckpt_splats["scales"]),
            "quats": torch.nn.Parameter(ckpt_splats["quats"]),
            "opacities": torch.nn.Parameter(ckpt_splats["opacities"]),
            "sh0": torch.nn.Parameter(ckpt_splats["sh0"]),
            "shN": torch.nn.Parameter(ckpt_splats["shN"]),
            "times": torch.nn.Parameter(ckpt_splats["times"]),
            "durations": torch.nn.Parameter(ckpt_splats["durations"]),
            "velocities": torch.nn.Parameter(ckpt_splats["velocities"]),
        }).to(self.device)
        print(f"  Loaded {len(self.splats['means']):,} Gaussians")

        # Set start step for resume
        self.start_step = ckpt.get("step", 0) + 1
        print(f"  Will resume from step {self.start_step}")

        # Resize gradient accumulator to match loaded Gaussians
        self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)
        self.grad_count = 0

        # Reinitialize optimizers for the new splats (needed for resume training)
        if not self.cfg.export_only:
            cfg = self.cfg
            self.optimizers = {
                "means": SelectiveAdam([{"params": self.splats["means"], "lr": cfg.position_lr, "name": "means"}], eps=1e-15),
                "scales": SelectiveAdam([{"params": self.splats["scales"], "lr": cfg.scales_lr, "name": "scales"}], eps=1e-15),
                "quats": SelectiveAdam([{"params": self.splats["quats"], "lr": cfg.quats_lr, "name": "quats"}], eps=1e-15),
                "opacities": SelectiveAdam([{"params": self.splats["opacities"], "lr": cfg.opacities_lr, "name": "opacities"}], eps=1e-15),
                "sh0": SelectiveAdam([{"params": self.splats["sh0"], "lr": cfg.sh0_lr, "name": "sh0"}], eps=1e-15),
                "shN": SelectiveAdam([{"params": self.splats["shN"], "lr": cfg.shN_lr, "name": "shN"}], eps=1e-15),
                "times": SelectiveAdam([{"params": self.splats["times"], "lr": cfg.times_lr, "name": "times"}], eps=1e-15),
                "durations": SelectiveAdam([{"params": self.splats["durations"], "lr": cfg.durations_lr, "name": "durations"}], eps=1e-15),
                "velocities": SelectiveAdam([{"params": self.splats["velocities"], "lr": cfg.velocity_lr_start, "name": "velocities"}], eps=1e-15),
            }

            # Load optimizer states if available
            if "optimizers" in ckpt:
                for name, opt_state in ckpt["optimizers"].items():
                    if name in self.optimizers:
                        try:
                            self.optimizers[name].load_state_dict(opt_state)
                        except Exception as e:
                            print(f"  Warning: Could not load optimizer state for {name}: {e}")
                print("  Loaded optimizer states")

            # Re-initialize strategy state for the loaded Gaussians
            self.cfg.strategy.check_sanity(self.splats, self.optimizers)
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.strategy_state = self.cfg.strategy.initialize_state()

    def export_from_checkpoint(self):
        """Export PLY sequence and videos from loaded checkpoint (no training)."""
        if self.cfg.ckpt_path is None:
            raise ValueError("No checkpoint path provided for export_only mode")

        step = self.start_step - 1  # The step at which checkpoint was saved
        print(f"\n[Export] Exporting from checkpoint at step {step}")

        if self.world_rank == 0:
            # Render trajectory videos
            self.render_traj(step=step)

            # Export PLY sequence
            if self.cfg.export_ply:
                self.export_ply_sequence(step=step)

        print("[Export] Complete!")

    def _turbo_colormap(self, values: Tensor) -> Tensor:
        """
        Apply turbo colormap to normalized values [0, 1].

        Args:
            values: [N] tensor with values in [0, 1]
        Returns:
            colors: [N, 3] RGB colors
        """
        # Turbo colormap approximation (attempt to match matplotlib's turbo)
        # Based on: https://gist.github.com/mikhailov-work/0d177465a8151eb6edd1768d07d17c74
        x = values.unsqueeze(-1)  # [N, 1]

        # Red channel
        r = (0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))))
        # Green channel
        g = (0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))))
        # Blue channel
        b = (0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))))

        colors = torch.cat([r, g, b], dim=-1).clamp(0, 1)  # [N, 3]
        return colors

    def _add_colorbar(self, img: Tensor, vmin: float, vmax: float, label: str = "", bar_width: int = 40) -> Tensor:
        """
        Add a vertical colorbar to the right side of an image.

        Args:
            img: [3, H, W] image tensor
            vmin: minimum value for colorbar
            vmax: maximum value for colorbar
            label: label text for the colorbar
            bar_width: width of the colorbar in pixels

        Returns:
            img_with_bar: [3, H, W + bar_width + margin] image with colorbar
        """
        C, H, W = img.shape
        device = img.device

        # Create colorbar gradient (bottom=0/blue to top=1/red)
        gradient = torch.linspace(0, 1, H, device=device)  # [H] from 0 to 1
        gradient = gradient.flip(0)  # Flip so top is high value (red)
        gradient = gradient.unsqueeze(0).unsqueeze(-1).expand(1, H, bar_width)  # [1, H, bar_width]

        # Apply turbo colormap to gradient
        gradient_flat = gradient.reshape(-1)  # [H * bar_width]
        colorbar_colors = self._turbo_colormap(gradient_flat)  # [H * bar_width, 3]
        colorbar = colorbar_colors.reshape(H, bar_width, 3).permute(2, 0, 1)  # [3, H, bar_width]

        # Add black border around colorbar
        colorbar[:, 0:2, :] = 0  # Top border
        colorbar[:, -2:, :] = 0  # Bottom border
        colorbar[:, :, 0:2] = 0  # Left border
        colorbar[:, :, -2:] = 0  # Right border

        # Create margin (white space between image and colorbar)
        margin_width = 10
        margin = torch.ones(3, H, margin_width, device=device)

        # Add tick marks on the margin (pointing to colorbar)
        tick_length = 6
        # Top tick (max value) - red
        margin[:, 8:12, -tick_length:] = 0
        # Bottom tick (min value) - blue
        margin[:, H-12:H-8, -tick_length:] = 0
        # Middle tick
        margin[:, H//2-2:H//2+2, -tick_length:] = 0
        # Quarter ticks
        margin[:, H//4-1:H//4+1, -tick_length+2:] = 0
        margin[:, 3*H//4-1:3*H//4+1, -tick_length+2:] = 0

        # Concatenate: [img | margin | colorbar]
        img_with_bar = torch.cat([img, margin, colorbar], dim=2)

        return img_with_bar, vmin, vmax

    def render_duration_velocity_images(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        t: float,
        static_mode: bool = False,
        use_temporal_opacity: bool = True,
        add_colorbar: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Render duration and velocity visualization images using turbo heatmap.

        Args:
            add_colorbar: If True, add colorbar with min/max values to the right side

        Returns:
            duration_img: [3, H, W(+colorbar)] - duration heatmap (blue=low, red=high)
            velocity_img: [3, H, W(+colorbar)] - velocity magnitude heatmap (blue=low, red=high)
        """
        # Get positions at time t
        if static_mode:
            means = self.splats["means"]
        else:
            means = self.compute_positions_at_time(t)

        # Get temporal opacity
        if use_temporal_opacity:
            temporal_opacity = self.compute_temporal_opacity(t)
        else:
            temporal_opacity = torch.ones(len(means), device=self.device)

        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        base_opacity = torch.sigmoid(self.splats["opacities"])
        opacities = torch.clamp(base_opacity * temporal_opacity, min=1e-4)

        # --- Duration visualization ---
        # Get durations and normalize to [0, 1]
        durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)  # [N]
        dur_min, dur_max = durations_exp.min().item(), durations_exp.max().item()
        dur_normalized = (durations_exp - dur_min) / (dur_max - dur_min + 1e-8)  # [N]

        # Apply turbo heatmap colormap
        dur_colors = self._turbo_colormap(dur_normalized)  # [N, 3]

        # Render duration image (sh_degree=None means colors are direct RGB)
        dur_render, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=dur_colors,  # [N, 3]
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,  # Direct RGB colors
            packed=self.cfg.packed,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        duration_img = dur_render[0, ..., :3].clamp(0, 1).permute(2, 0, 1)  # [3, H, W]

        # --- Velocity visualization ---
        # Get velocity magnitude and normalize to [0, 1]
        vel_mag = self.splats["velocities"].norm(dim=-1)  # [N]
        vel_min, vel_max = vel_mag.min().item(), vel_mag.max().item()
        vel_normalized = (vel_mag - vel_min) / (vel_max - vel_min + 1e-8)  # [N]

        # Apply turbo heatmap colormap
        vel_colors = self._turbo_colormap(vel_normalized)  # [N, 3]

        # Render velocity image
        vel_render, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=vel_colors,  # [N, 3]
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,  # Direct RGB colors
            packed=self.cfg.packed,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        velocity_img = vel_render[0, ..., :3].clamp(0, 1).permute(2, 0, 1)  # [3, H, W]

        # Add colorbars if requested
        if add_colorbar:
            duration_img, _, _ = self._add_colorbar(duration_img, dur_min, dur_max, "Duration")
            velocity_img, _, _ = self._add_colorbar(velocity_img, vel_min, vel_max, "Velocity")

        return duration_img, velocity_img

    def compute_temporal_opacity(self, t: float) -> Tensor:
        """
        Temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)^2)

        Paper equation for temporal Gaussian distribution.
        """
        mu_t = self.splats["times"]  # [N, 1]
        s = torch.exp(self.splats["durations"])  # [N, 1] - duration in original space
        return torch.exp(-0.5 * ((t - mu_t) / (s + 1e-8)) ** 2).squeeze(-1)  # [N]

    def compute_positions_at_time(self, t: float) -> Tensor:
        """
        Position at time t: µx(t) = µx + v · (t - µt)

        Paper equation for linear velocity model.
        """
        mu_x = self.splats["means"]  # [N, 3]
        mu_t = self.splats["times"]  # [N, 1]
        v = self.splats["velocities"]  # [N, 3]
        return mu_x + v * (t - mu_t)  # [N, 3]

    def compute_4d_regularization(self, temporal_opacity: Tensor) -> Tensor:
        """
        4D Regularization: Lreg(t) = (1/N) * Σ(σ * sg[σ(t)])

        From paper: Uses stop-gradient on temporal opacity to prevent minimizing it.
        This encourages Gaussians to have high opacity at their canonical time
        without collapsing temporal opacity to zero.
        """
        base_opacity = torch.sigmoid(self.splats["opacities"])  # [N]
        # Stop-gradient on temporal opacity - only backprop through base opacity
        temporal_opa_sg = temporal_opacity.detach()  # sg[σ(t)]
        # Regularization: mean of (σ * sg[σ(t)])
        reg = (base_opacity * temporal_opa_sg).mean()
        return reg

    def compute_difficulty_scores(self) -> Tensor:
        """
        Compute per-Gaussian difficulty scores based on velocity and duration.

        Difficulty = α * normalized_velocity + β * normalized_inverse_duration

        High velocity + low duration = high difficulty (fast-moving, short-lived regions)
        These regions need more attention during training.

        Returns:
            difficulty: [N] tensor of difficulty scores in range [1, motion_weight_max]
        """
        cfg = self.cfg
        with torch.no_grad():
            # Velocity magnitude
            velocities = self.splats["velocities"]  # [N, 3]
            velocity_mag = torch.norm(velocities, dim=-1)  # [N]

            # Duration (original space)
            durations = torch.exp(self.splats["durations"]).squeeze(-1)  # [N]

            # Inverse duration (higher for short-lived Gaussians)
            inv_duration = 1.0 / (durations + 1e-6)

            # Normalize to [0, 1] range using percentiles to handle outliers
            vel_min = velocity_mag.min()
            vel_max = velocity_mag.quantile(0.95).clamp(min=vel_min + 1e-6)
            velocity_norm = ((velocity_mag - vel_min) / (vel_max - vel_min + 1e-6)).clamp(0, 1)

            inv_dur_min = inv_duration.min()
            inv_dur_max = inv_duration.quantile(0.95).clamp(min=inv_dur_min + 1e-6)
            inv_duration_norm = ((inv_duration - inv_dur_min) / (inv_dur_max - inv_dur_min + 1e-6)).clamp(0, 1)

            # Combined difficulty score
            difficulty_raw = (
                cfg.motion_weight_velocity * velocity_norm +
                cfg.motion_weight_duration * inv_duration_norm
            )

            # Scale to [1, motion_weight_max] range
            difficulty = 1.0 + difficulty_raw * (cfg.motion_weight_max - 1.0)

        return difficulty

    def render_difficulty_map(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        t: float,
        difficulty: Tensor,
        static_mode: bool = False,
        use_temporal_opacity: bool = True,
    ) -> Tensor:
        """
        Render a per-pixel difficulty map by rasterizing Gaussian difficulty scores.

        Args:
            difficulty: [N] per-Gaussian difficulty scores

        Returns:
            difficulty_map: [B, H, W, 1] per-pixel difficulty weights
        """
        # Position: apply velocity during 4D phase
        if static_mode:
            means = self.splats["means"]
        else:
            means = self.compute_positions_at_time(t)

        # Temporal opacity: skip during warmup
        if use_temporal_opacity:
            temporal_opacity = self.compute_temporal_opacity(t)
        else:
            temporal_opacity = torch.ones(len(means), device=self.device)

        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        base_opacity = torch.sigmoid(self.splats["opacities"])
        opacities = base_opacity * temporal_opacity
        opacities = torch.clamp(opacities, min=1e-4)

        # Use difficulty as the "color" for rasterization
        difficulty_colors = difficulty.unsqueeze(-1).expand(-1, 3)  # [N, 3]

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        difficulty_render, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=difficulty_colors,  # [N, 3] difficulty as color
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            sh_degree=None,  # No SH, direct color
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            rasterize_mode=rasterize_mode,
        )

        # Take one channel (all 3 are the same)
        difficulty_map = difficulty_render[..., :1]  # [B, H, W, 1]

        # Normalize: background pixels get weight 1.0
        # difficulty_map is 0 where no Gaussians rendered
        difficulty_map = torch.where(
            difficulty_map < 0.5,  # Background (no Gaussians)
            torch.ones_like(difficulty_map),
            difficulty_map
        )

        return difficulty_map

    def temporal_coverage_densification(self, step: int) -> int:
        """
        Temporal coverage densification: spawn new Gaussians in regions where
        temporal coverage is poor (high velocity + low duration).

        The idea:
        1. Find Gaussians with high velocity and low duration
        2. These Gaussians "disappear" at certain times, leaving gaps
        3. Spawn new Gaussians at adjacent time points to fill gaps

        Returns:
            Number of new Gaussians spawned
        """
        cfg = self.cfg

        with torch.no_grad():
            velocities = self.splats["velocities"]  # [N, 3]
            durations = torch.exp(self.splats["durations"]).squeeze(-1)  # [N]
            times = self.splats["times"].squeeze(-1)  # [N]
            base_opacity = torch.sigmoid(self.splats["opacities"])  # [N]
            velocity_mag = torch.norm(velocities, dim=-1)  # [N]

            # Find candidates: high velocity, low duration, decent opacity
            candidate_mask = (
                (velocity_mag > cfg.temporal_densify_velocity_threshold) &
                (durations < cfg.temporal_densify_duration_threshold) &
                (base_opacity > cfg.temporal_densify_opacity_threshold)
            )

            n_candidates = candidate_mask.sum().item()
            if n_candidates == 0:
                return 0

            # Limit number of spawns
            n_spawn = min(n_candidates, cfg.temporal_densify_max_spawn)

            # Score candidates by velocity * inverse_duration (higher = more important)
            candidate_idx = candidate_mask.nonzero(as_tuple=True)[0]
            scores = velocity_mag[candidate_idx] / (durations[candidate_idx] + 1e-6)

            # Select top candidates
            if len(candidate_idx) > n_spawn:
                _, top_indices = scores.topk(n_spawn)
                selected_idx = candidate_idx[top_indices]
            else:
                selected_idx = candidate_idx

            n_to_spawn = len(selected_idx)
            if n_to_spawn == 0:
                return 0

            # For each selected Gaussian, spawn a new one at a nearby time
            # Strategy: spawn at time = original_time ± duration (fill the gap)
            spawn_direction = torch.sign(torch.randn(n_to_spawn, device=self.device))
            time_offset = durations[selected_idx] * spawn_direction * 1.5  # Spawn 1.5 durations away

            new_times = (times[selected_idx] + time_offset).clamp(0, 1)

            # Compute positions at the new times (using velocity)
            new_means = (
                self.splats["means"][selected_idx] +
                self.splats["velocities"][selected_idx] * (new_times - times[selected_idx]).unsqueeze(-1)
            )

            # Copy other parameters from parent Gaussians
            new_quats = self.splats["quats"][selected_idx].clone()
            new_scales = self.splats["scales"][selected_idx].clone()
            new_opacities = self.splats["opacities"][selected_idx].clone()
            new_sh0 = self.splats["sh0"][selected_idx].clone()
            new_shN = self.splats["shN"][selected_idx].clone()

            # New Gaussians get fresh duration (inherit from parent but reset)
            new_durations = self.splats["durations"][selected_idx].clone()

            # New Gaussians get reduced velocity (they're filling gaps, not moving as fast)
            new_velocities = self.splats["velocities"][selected_idx].clone() * 0.5

        # Concatenate new Gaussians to existing parameters using _update_param_with_optimizer
        # This properly handles optimizer state extension
        param_names = ["means", "quats", "scales", "opacities", "sh0", "shN", "times", "durations", "velocities"]
        new_values_dict = {
            "means": new_means,
            "quats": new_quats,
            "scales": new_scales,
            "opacities": new_opacities,
            "sh0": new_sh0,
            "shN": new_shN,
            "times": new_times.unsqueeze(-1),
            "durations": new_durations,
            "velocities": new_velocities,
        }

        for name in param_names:
            old_param = self.splats[name]
            new_values = new_values_dict[name]
            new_param = torch.cat([old_param.data, new_values], dim=0)

            # Use the gsplat helper to update param and optimizer state
            _update_param_with_optimizer(
                param_fn=lambda p, n=new_param: n,
                optimizer=self.optimizers[name],
                param=old_param,
                name=name,
            )
            self.splats[name] = self.optimizers[name].param_groups[0]["params"][0]

        # Reinitialize strategy state (the strategy tracks per-Gaussian info)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()

        # Resize gradient accumulator
        self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)
        self.grad_count = 0

        return n_to_spawn

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        t: float,
        sh_degree: int,
        static_mode: bool = False,
        use_temporal_opacity: bool = True,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize 4D Gaussians at time t.

        Args:
            static_mode: If True, don't apply velocities (canonical positions).
            use_temporal_opacity: If True, compute temporal opacity.
        """
        # Position: apply velocity during 4D phase
        if static_mode:
            means = self.splats["means"]
        else:
            means = self.compute_positions_at_time(t)

        # Temporal opacity: skip during warmup
        if use_temporal_opacity:
            temporal_opacity = self.compute_temporal_opacity(t)
        else:
            temporal_opacity = torch.ones(len(means), device=self.device)

        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        base_opacity = torch.sigmoid(self.splats["opacities"])

        # Combined opacity: σ(x,t) = σ(t) × σ
        opacities = base_opacity * temporal_opacity
        opacities = torch.clamp(opacities, min=1e-4)  # Prevent black dots

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], dim=1)

        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        renders, alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            sh_degree=sh_degree,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        info["temporal_opacity"] = temporal_opacity
        info["positions_at_t"] = means
        return renders, alphas, info

    def relocate_gaussians(self, step: int) -> int:
        """
        Periodic relocation of low-opacity Gaussians (from paper).

        Sampling score: s = λg·∇g + λo·σ
        - Dead Gaussians (low opacity) are relocated to high-score regions.
        - This is independent of DefaultStrategy/MCMCStrategy.
        """
        cfg = self.cfg

        with torch.no_grad():
            base_opacity = torch.sigmoid(self.splats["opacities"])
            n_total = len(base_opacity)

            # Find dead Gaussians (low base opacity)
            dead_mask = base_opacity < cfg.relocation_opacity_threshold
            n_dead_total = dead_mask.sum().item()

            if n_dead_total == 0:
                return 0

            alive_mask = ~dead_mask
            if alive_mask.sum() == 0:
                return 0

            # Cap the number of relocations to prevent scene darkening
            max_relocate = int(n_total * cfg.relocation_max_ratio)
            n_to_relocate = min(n_dead_total, max_relocate)

            if n_to_relocate == 0:
                return 0

            dead_idx_all = dead_mask.nonzero(as_tuple=True)[0]
            alive_idx = alive_mask.nonzero(as_tuple=True)[0]

            # If we have more dead than we can relocate, pick the lowest opacity ones
            if n_dead_total > n_to_relocate:
                dead_opacities = base_opacity[dead_idx_all]
                _, sorted_indices = dead_opacities.sort()
                dead_idx = dead_idx_all[sorted_indices[:n_to_relocate]]
            else:
                dead_idx = dead_idx_all

            n_dead = len(dead_idx)

            # Compute sampling score: s = λg·∇g + λo·σ
            # Normalize gradient accumulator
            if self.grad_count > 0:
                grad_score = self.grad_accum / self.grad_count
                grad_score = grad_score / (grad_score.max() + 1e-8)  # Normalize to [0, 1]
            else:
                grad_score = torch.zeros_like(base_opacity)

            # Sampling score for alive Gaussians
            alive_grad = grad_score[alive_idx]
            alive_opa = base_opacity[alive_idx]
            alive_opa_norm = alive_opa / (alive_opa.max() + 1e-8)

            sampling_score = cfg.relocation_lambda_grad * alive_grad + cfg.relocation_lambda_opa * alive_opa_norm
            sampling_score = sampling_score.clamp(min=1e-8)

            # Sample sources weighted by sampling score
            probs = sampling_score / sampling_score.sum()
            sampled = torch.multinomial(probs, n_dead, replacement=True)
            source_idx = alive_idx[sampled]

            # Copy parameters from source to dead
            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "means":
                    # Add small noise to positions
                    noise = torch.randn(n_dead, 3, device=self.device) * 0.01 * self.scene_scale
                    p[dead_idx] = p[source_idx] + noise
                elif name == "opacities":
                    # Reset opacity to slightly below source to let it learn
                    source_opa = torch.sigmoid(p[source_idx])
                    # Start at 80% of source opacity instead of fixed init_opacity
                    new_opa = source_opa * 0.8
                    p[dead_idx] = torch.logit(new_opa.clamp(0.01, 0.99))
                else:
                    # Copy other parameters
                    p[dead_idx] = p[source_idx]
                return torch.nn.Parameter(p, requires_grad=p.requires_grad)

            def opt_fn(key: str, v: Tensor) -> Tensor:
                # Reset optimizer state for relocated Gaussians
                v[dead_idx] = 0
                return v

            _update_param_with_optimizer(param_fn, opt_fn, self.splats, self.optimizers)

            # Only reset gradient for relocated Gaussians, not all
            self.grad_accum[dead_idx] = 0
            # Don't reset grad_count - keep accumulating

        return n_dead

    def prune_gaussians(self, step: int) -> int:
        """Prune low-opacity Gaussians during canonical phase."""
        cfg = self.cfg

        with torch.no_grad():
            base_opacity = torch.sigmoid(self.splats["opacities"])

            # Find Gaussians to prune
            prune_mask = base_opacity < cfg.prune_opacity_threshold
            n_prune = prune_mask.sum().item()

            if n_prune == 0:
                return 0

            # Don't prune all Gaussians
            n_remaining = (~prune_mask).sum().item()
            if n_remaining < 1000:
                print(f"[Prune] Would leave only {n_remaining} Gaussians, skipping")
                return 0

            # Remove pruned Gaussians
            remove(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mask=prune_mask,
            )

            # Resize gradient accumulator
            self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)

        return n_prune

    def train(self):
        cfg = self.cfg
        device = self.device

        # Save config
        if self.world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f, default_flow_style=False)

        max_steps = cfg.max_steps
        canonical_end = cfg.warmup_steps + cfg.canonical_phase_steps

        # Learning rate schedulers
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]

        # Data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, persistent_workers=True, pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        print("\n" + "="*70)
        print("[FreeTime4D] Training Phases:")
        print(f"  Warmup: steps 0-{cfg.warmup_steps}")
        print(f"    - Basic appearance learning (colors, scales, opacity)")
        print(f"    - No temporal opacity (all Gaussians visible)")
        print(f"    - All temporal params frozen (times, durations, velocities)")
        print(f"  Canonical: steps {cfg.warmup_steps}-{canonical_end}")
        print(f"    - Static positions (velocities not applied)")
        print(f"    - Temporal opacity active (learn visibility width)")
        print(f"    - Only durations optimized (times FROZEN to preserve velocity relationship!)")
        print(f"    - Pruning: every {cfg.prune_every} steps")
        print(f"  Full 4D: steps {canonical_end}+")
        print(f"    - Motion enabled: µx(t) = µx + v·(t-µt)")
        print(f"    - All temporal params optimized (times, durations, velocities)")
        print(f"    - 4D regularization: Lreg = (1/N)Σ(σ·sg[σ(t)]), λ={cfg.lambda_4d_reg}")
        print(f"    - Relocation: every {cfg.relocation_every} steps with score s = λg·∇g + λo·σ")
        print("="*70 + "\n")

        # Resume from checkpoint if applicable
        start_step = self.start_step
        if start_step > 0:
            print(f"[Resume] Resuming training from step {start_step}")

        global_tic = time.time()
        pbar = tqdm.tqdm(range(start_step, max_steps))

        for step in pbar:
            # Load batch
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            t = data["time"].to(device).mean().item()

            # Phase determination
            in_warmup = step < cfg.warmup_steps
            in_canonical = cfg.warmup_steps <= step < canonical_end
            in_4d = step >= canonical_end

            # Static mode during warmup and canonical
            static_mode = in_warmup or in_canonical
            use_temporal_opacity = not in_warmup

            # Adjust learning rates for temporal parameters
            if in_warmup:
                # Freeze all temporal params during warmup
                for name in ["times", "durations", "velocities"]:
                    for pg in self.optimizers[name].param_groups:
                        pg["lr"] = 0.0
            elif in_canonical:
                # IMPORTANT: Freeze both times AND velocities to preserve their relationship!
                # The velocities from RoMa were estimated at the original times.
                # If we optimize times without velocities, the time-velocity relationship breaks.
                # Only optimize durations (temporal window width) during canonical.
                for pg in self.optimizers["velocities"].param_groups:
                    pg["lr"] = 0.0
                for pg in self.optimizers["times"].param_groups:
                    pg["lr"] = 0.0  # FREEZE times to preserve velocity relationship
                for pg in self.optimizers["durations"].param_groups:
                    pg["lr"] = cfg.durations_lr * math.sqrt(cfg.batch_size)
            else:  # in_4d
                # Velocity LR annealing
                progress = (step - canonical_end) / max(max_steps - canonical_end, 1)
                vel_lr = cfg.velocity_lr_start * (cfg.velocity_lr_end / cfg.velocity_lr_start) ** progress
                for pg in self.optimizers["velocities"].param_groups:
                    pg["lr"] = vel_lr * math.sqrt(cfg.batch_size)
                for pg in self.optimizers["times"].param_groups:
                    pg["lr"] = cfg.times_lr * math.sqrt(cfg.batch_size)
                for pg in self.optimizers["durations"].param_groups:
                    pg["lr"] = cfg.durations_lr * math.sqrt(cfg.batch_size)

            # SH degree schedule
            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Forward pass
            renders, alphas, info = self.rasterize_splats(
                camtoworlds, Ks, width, height, t, sh_degree,
                static_mode=static_mode, use_temporal_opacity=use_temporal_opacity,
            )
            colors = renders[..., :3]

            # Strategy pre-backward (for gradient accumulation)
            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # ============================================================
            # LOSS COMPUTATION (Paper: L = λimg*L1 + λssim*SSIM + λperc*LPIPS + λreg*Lreg)
            # ============================================================

            # Clamp colors to [0, 1] for loss computation (rendering can produce values slightly > 1)
            colors = torch.clamp(colors, 0.0, 1.0)

            colors_p = colors.permute(0, 3, 1, 2)  # [B, 3, H, W]
            pixels_p = pixels.permute(0, 3, 1, 2)  # [B, 3, H, W]

            # ============================================================
            # Motion-Weighted Loss: Weight loss higher for difficult regions
            # (high velocity + low duration = fast-moving, transient regions)
            # ============================================================
            use_motion_weight = (
                cfg.use_motion_weighted_loss and
                in_4d and
                step >= cfg.motion_weight_start_iter
            )

            if use_motion_weight:
                # Compute per-Gaussian difficulty scores
                difficulty = self.compute_difficulty_scores()

                # Render difficulty map (per-pixel weights)
                difficulty_map = self.render_difficulty_map(
                    camtoworlds, Ks, width, height, t, difficulty,
                    static_mode=static_mode, use_temporal_opacity=use_temporal_opacity,
                )  # [B, H, W, 1]

                # Expand to match color channels
                weight_map = difficulty_map.expand(-1, -1, -1, 3)  # [B, H, W, 3]
                weight_map_p = weight_map.permute(0, 3, 1, 2)  # [B, 3, H, W]
            else:
                weight_map = None
                weight_map_p = None

            # 1. L1 Loss (reconstruction) - optionally weighted
            if use_motion_weight and weight_map is not None:
                # Per-pixel weighted L1 loss
                l1_per_pixel = torch.abs(colors - pixels)  # [B, H, W, 3]
                l1_loss = (l1_per_pixel * weight_map).mean() / weight_map.mean()
            else:
                l1_loss = F.l1_loss(colors, pixels)

            # 2. SSIM Loss (structural similarity)
            ssim_val = fused_ssim(colors_p, pixels_p, padding="valid")
            ssim_loss = 1.0 - ssim_val

            # 3. LPIPS Loss (perceptual similarity) - paper: λperc=0.01
            lpips_loss = self.lpips(colors_p, pixels_p) if cfg.lambda_perc > 0 else torch.tensor(0.0, device=device)

            # 4. 4D Regularization (paper: λreg=1e-2) - only during 4D phase
            # Lreg(t) = (1/N) * Σ(σ * sg[σ(t)]) - stop-gradient on temporal opacity
            reg_4d_loss = torch.tensor(0.0, device=device)
            duration_reg_loss = torch.tensor(0.0, device=device)
            if in_4d and cfg.lambda_4d_reg > 0:
                temporal_opacity = info["temporal_opacity"]
                reg_4d_loss = self.compute_4d_regularization(temporal_opacity)

                # Duration regularization: penalize wide temporal windows (cause blur)
                # This helps dynamic regions have sharper temporal profiles
                if cfg.lambda_duration_reg > 0:
                    durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)
                    # Penalize durations larger than target
                    target_duration = cfg.init_duration  # 0.1
                    excess = torch.clamp(durations_exp - target_duration, min=0)
                    duration_reg_loss = (excess ** 2).mean()

            # Combine all losses with paper weights
            # Paper: λimg=0.8, λssim=0.2, λperc=0.01, λreg=1e-2
            loss_img = cfg.lambda_img * l1_loss
            loss_ssim = cfg.lambda_ssim * ssim_loss
            loss_lpips = cfg.lambda_perc * lpips_loss
            loss_4d_reg = cfg.lambda_4d_reg * reg_4d_loss
            loss_dur_reg = cfg.lambda_duration_reg * duration_reg_loss

            # Total loss
            loss = loss_img + loss_ssim + loss_lpips + loss_4d_reg + loss_dur_reg

            # Backward
            loss.backward()

            # Accumulate gradients for relocation sampling score
            if in_4d and "means" in self.splats and self.splats["means"].grad is not None:
                grad_mag = self.splats["means"].grad.norm(dim=-1)
                if len(self.grad_accum) == len(grad_mag):
                    self.grad_accum += grad_mag
                    self.grad_count += 1

            # Optimizer step
            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            for sched in schedulers:
                sched.step()

            # Strategy post-backward (DefaultStrategy/MCMCStrategy operations)
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )

            # Resize gradient accumulator if strategy changed Gaussian count
            if len(self.grad_accum) != len(self.splats["means"]):
                self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)
                self.grad_count = 0

            # Pruning (independent of phase, uses config start/stop)
            if cfg.use_pruning:
                if cfg.prune_start_iter <= step < cfg.prune_stop_iter:
                    if step % cfg.prune_every == 0:
                        n_pruned = self.prune_gaussians(step)
                        if n_pruned > 0:
                            print(f"[Prune] Step {step}: removed {n_pruned}, remaining {len(self.splats['means'])}")

            # Periodic relocation during 4D phase (independent of strategy)
            if cfg.use_relocation and in_4d:
                if cfg.relocation_start_iter <= step < cfg.relocation_stop_iter:
                    if step % cfg.relocation_every == 0:
                        n_relocated = self.relocate_gaussians(step)
                        if n_relocated > 0:
                            print(f"[Relocation] Step {step}: relocated {n_relocated}")

            # Temporal coverage densification: spawn new Gaussians in
            # regions with high velocity + low duration (poor temporal coverage)
            if cfg.use_temporal_densification and in_4d:
                if cfg.temporal_densify_start_iter <= step < cfg.temporal_densify_stop_iter:
                    if step % cfg.temporal_densify_every == 0:
                        n_spawned = self.temporal_coverage_densification(step)
                        if n_spawned > 0:
                            print(f"[TemporalDensify] Step {step}: spawned {n_spawned} new Gaussians for temporal coverage")

            # Progress bar
            phase = "WARM" if in_warmup else ("CANON" if in_canonical else "4D")
            motion_marker = "+MW" if use_motion_weight else ""
            pbar.set_description(
                f"[{phase}{motion_marker}] loss={loss.item():.4f} l1={l1_loss.item():.4f} "
                f"t={t:.2f} N={len(self.splats['means'])}"
            )

            # ============================================================
            # TENSORBOARD LOGGING
            # ============================================================
            if self.world_rank == 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3

                # --- Total Loss ---
                self.writer.add_scalar("loss/total", loss.item(), step)

                # --- Individual Loss Components (raw values) ---
                self.writer.add_scalar("loss/l1_raw", l1_loss.item(), step)
                self.writer.add_scalar("loss/ssim_raw", ssim_loss.item(), step)
                self.writer.add_scalar("loss/lpips_raw", lpips_loss.item(), step)
                self.writer.add_scalar("loss/4d_reg_raw", reg_4d_loss.item(), step)

                # --- Weighted Loss Components (what goes into total) ---
                self.writer.add_scalar("loss_weighted/l1", loss_img.item(), step)
                self.writer.add_scalar("loss_weighted/ssim", loss_ssim.item(), step)
                self.writer.add_scalar("loss_weighted/lpips", loss_lpips.item(), step)
                self.writer.add_scalar("loss_weighted/4d_reg", loss_4d_reg.item(), step)

                # --- Quality Metrics ---
                self.writer.add_scalar("metrics/ssim", ssim_val.item(), step)  # Higher is better
                self.writer.add_scalar("metrics/psnr", -10 * torch.log10(F.mse_loss(colors, pixels)).item(), step)

                # --- Gaussian Statistics ---
                self.writer.add_scalar("gaussians/count", len(self.splats["means"]), step)
                self.writer.add_scalar("gaussians/mem_gb", mem, step)

                with torch.no_grad():
                    base_opacity = torch.sigmoid(self.splats["opacities"])
                    self.writer.add_scalar("gaussians/opacity_mean", base_opacity.mean().item(), step)
                    self.writer.add_scalar("gaussians/opacity_min", base_opacity.min().item(), step)
                    self.writer.add_scalar("gaussians/opacity_max", base_opacity.max().item(), step)

                    scales_exp = torch.exp(self.splats["scales"])
                    self.writer.add_scalar("gaussians/scale_mean", scales_exp.mean().item(), step)

                # --- Temporal Statistics ---
                with torch.no_grad():
                    temporal_opa = info["temporal_opacity"]
                    self.writer.add_scalar("temporal/opacity_mean", temporal_opa.mean().item(), step)
                    self.writer.add_scalar("temporal/opacity_min", temporal_opa.min().item(), step)
                    self.writer.add_scalar("temporal/opacity_max", temporal_opa.max().item(), step)

                    durations_exp = torch.exp(self.splats["durations"])
                    self.writer.add_scalar("temporal/duration_mean", durations_exp.mean().item(), step)
                    self.writer.add_scalar("temporal/duration_min", durations_exp.min().item(), step)
                    self.writer.add_scalar("temporal/duration_max", durations_exp.max().item(), step)

                    times = self.splats["times"]
                    self.writer.add_scalar("temporal/time_mean", times.mean().item(), step)
                    self.writer.add_scalar("temporal/time_std", times.std().item(), step)

                    vel_mag = self.splats["velocities"].norm(dim=-1)
                    self.writer.add_scalar("temporal/velocity_mean", vel_mag.mean().item(), step)
                    self.writer.add_scalar("temporal/velocity_max", vel_mag.max().item(), step)
                    self.writer.add_scalar("temporal/velocity_min", vel_mag.min().item(), step)

                # --- Motion-Weighted Loss Statistics ---
                if use_motion_weight:
                    difficulty = self.compute_difficulty_scores()
                    self.writer.add_scalar("motion_weight/difficulty_mean", difficulty.mean().item(), step)
                    self.writer.add_scalar("motion_weight/difficulty_max", difficulty.max().item(), step)
                    self.writer.add_scalar("motion_weight/difficulty_min", difficulty.min().item(), step)
                    # Count high-difficulty Gaussians (> 2x weight)
                    n_high_difficulty = (difficulty > 2.0).sum().item()
                    self.writer.add_scalar("motion_weight/n_high_difficulty", n_high_difficulty, step)
                    self.writer.add_scalar("motion_weight/enabled", 1.0, step)
                else:
                    self.writer.add_scalar("motion_weight/enabled", 0.0, step)

                # --- Training Info ---
                self.writer.add_scalar("train/time_t", t, step)
                self.writer.add_scalar("train/sh_degree", sh_degree, step)
                self.writer.add_scalar("train/lr_means", self.optimizers["means"].param_groups[0]["lr"], step)
                self.writer.add_scalar("train/lr_velocities", self.optimizers["velocities"].param_groups[0]["lr"], step)

                # --- Phase Indicator (for visualization) ---
                phase_num = 0 if in_warmup else (1 if in_canonical else 2)
                self.writer.add_scalar("train/phase", phase_num, step)

            # --- Image Logging (every tb_image_every steps) ---
            if self.world_rank == 0 and step % cfg.tb_image_every == 0:
                with torch.no_grad():
                    # Ground truth vs Rendered side by side
                    gt_img = pixels[0].clamp(0, 1)  # [H, W, 3]
                    render_img = colors[0].detach().clamp(0, 1)  # [H, W, 3]

                    # Create side-by-side comparison: [GT | Rendered]
                    comparison = torch.cat([gt_img, render_img], dim=1)  # [H, 2*W, 3]
                    comparison = comparison.permute(2, 0, 1)  # [3, H, 2*W] for tensorboard

                    self.writer.add_image("images/gt_vs_render", comparison, step)

                    # Also log difference image (error visualization)
                    diff = (gt_img - render_img).abs()
                    diff_normalized = diff / (diff.max() + 1e-8)  # Normalize for visibility
                    diff_img = diff_normalized.permute(2, 0, 1)
                    self.writer.add_image("images/error_map", diff_img, step)

                    # Log histograms
                    if "temporal_opacity" in info:
                        temp_opa = info["temporal_opacity"]
                        self.writer.add_histogram("histograms/temporal_opacity", temp_opa, step)
                    self.writer.add_histogram("histograms/base_opacity", torch.sigmoid(self.splats["opacities"]), step)
                    self.writer.add_histogram("histograms/velocities", self.splats["velocities"].norm(dim=-1), step)
                    self.writer.add_histogram("histograms/durations", torch.exp(self.splats["durations"]), step)
                    self.writer.add_histogram("histograms/times", self.splats["times"], step)

            # --- Duration & Velocity Visualization (every 500 steps) ---
            if self.world_rank == 0 and step % 500 == 0 and step > 0:
                with torch.no_grad():
                    duration_img, velocity_img = self.render_duration_velocity_images(
                        camtoworlds, Ks, width, height, t,
                        static_mode=static_mode, use_temporal_opacity=use_temporal_opacity,
                    )
                    # duration_img: [3, H, W] - blue=short duration, red=long duration
                    # velocity_img: [3, H, W] - blue=slow, red=fast
                    self.writer.add_image("visualization/duration", duration_img, step)
                    self.writer.add_image("visualization/velocity", velocity_img, step)

                    # Log the min/max values to tensorboard for colorbar reference
                    durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)
                    vel_mag = self.splats["velocities"].norm(dim=-1)
                    self.writer.add_scalar("colorbar/duration_min", durations_exp.min().item(), step)
                    self.writer.add_scalar("colorbar/duration_max", durations_exp.max().item(), step)
                    self.writer.add_scalar("colorbar/velocity_min", vel_mag.min().item(), step)
                    self.writer.add_scalar("colorbar/velocity_max", vel_mag.max().item(), step)
                    print(f"[Vis] Step {step}: duration=[{durations_exp.min():.4f}, {durations_exp.max():.4f}], "
                          f"velocity=[{vel_mag.min():.4f}, {vel_mag.max():.4f}]")

            if self.world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.flush()

            # Save checkpoint, render trajectory, and export PLY at save_steps
            if step in [s - 1 for s in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print(f"[Save] Step {step}: {stats}")
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)

                data = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                    "optimizers": {k: v.state_dict() for k, v in self.optimizers.items()},
                }
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}.pt")

                # Render trajectory video at save steps
                if self.world_rank == 0:
                    self.render_traj(step=step)

                # Export PLY sequence at save steps
                if self.world_rank == 0 and cfg.export_ply:
                    self.export_ply_sequence(step=step)

            # Evaluation
            if step in [s - 1 for s in cfg.eval_steps]:
                self.eval(step)

        print(f"\n[Training] Complete! Total time: {time.time() - global_tic:.1f}s")

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluate on validation set (sampled for speed)."""
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False)
        metrics = defaultdict(list)
        ellipse_time = 0
        eval_count = 0

        # Sample every N-th frame for faster evaluation
        sample_every = cfg.eval_sample_every
        total_frames = len(valloader)
        eval_frames = total_frames // sample_every
        print(f"\n[Eval] Step {step}: evaluating {eval_frames} frames (every {sample_every}-th of {total_frames})")

        for i, data in enumerate(valloader):
            # Skip frames for faster evaluation
            if i % sample_every != 0:
                continue

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            t = data["time"].to(device).mean().item()
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            renders, _, _ = self.rasterize_splats(
                camtoworlds, Ks, width, height, t, cfg.sh_degree,
                static_mode=False, use_temporal_opacity=True,
            )
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(renders[..., :3], 0, 1)

            # Save image
            if self.world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(f"{self.render_dir}/{stage}_step{step}_{i:04d}.png", canvas)

                # Metrics
                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                eval_count += 1

        if self.world_rank == 0 and eval_count > 0:
            ellipse_time /= eval_count
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update({
                "ellipse_time": ellipse_time,
                "num_GS": len(self.splats["means"]),
            })
            print(
                f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} "
                f"Time: {ellipse_time:.3f}s/img, N: {stats['num_GS']}"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        """
        Render a 4D trajectory video with smooth camera motion and time progression.

        Creates a video that:
        1. Uses half the camera poses to generate a smooth trajectory
        2. Samples time smoothly from 0 to 1 over the video
        3. Exports as MP4 to result_dir/videos/
        """
        print("\n[Render Trajectory] Starting 4D trajectory rendering...")
        cfg = self.cfg
        device = self.device

        # Get camera poses from parser
        camtoworlds_all = self.parser.camtoworlds  # [N, 4, 4] numpy array

        # Use half the cameras for trajectory generation
        num_cams = len(camtoworlds_all)
        camtoworlds_subset = camtoworlds_all[:num_cams // 2]  # [N/2, 4, 4]

        # Extract [N, 3, 4] for trajectory generation functions
        camtoworlds_34 = camtoworlds_subset[:, :3, :]  # [N/2, 3, 4]

        n_frames = cfg.render_traj_n_frames

        # Generate trajectory
        if cfg.render_traj_path == "interp":
            # Interpolated smooth path through camera poses
            n_interp = max(1, n_frames // max(len(camtoworlds_34) - 1, 1))
            traj_poses = generate_interpolated_path(camtoworlds_34, n_interp)  # [M, 3, 4]
            # Subsample to exactly n_frames
            if len(traj_poses) > n_frames:
                indices = np.linspace(0, len(traj_poses) - 1, n_frames, dtype=int)
                traj_poses = traj_poses[indices]
            print(f"  Trajectory type: interpolated, {len(traj_poses)} frames")
        elif cfg.render_traj_path == "ellipse":
            # Elliptical path around the scene
            height = camtoworlds_34[:, 2, 3].mean()
            traj_poses = generate_ellipse_path_z(camtoworlds_34, n_frames=n_frames, height=height)
            print(f"  Trajectory type: ellipse, {len(traj_poses)} frames")
        else:
            raise ValueError(f"Unknown trajectory type: {cfg.render_traj_path}")

        # Convert to [N, 4, 4] by adding homogeneous row
        traj_poses_44 = np.concatenate([
            traj_poses,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(traj_poses), axis=0),
        ], axis=1)  # [N, 4, 4]

        traj_poses_44 = torch.from_numpy(traj_poses_44).float().to(device)

        # Get intrinsics from first camera
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height_img = list(self.parser.imsize_dict.values())[0]

        # Create video directory
        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)

        # Generate smooth time samples from 0 to 1
        time_samples = np.linspace(0, 1, len(traj_poses_44))

        print(f"  Rendering {len(traj_poses_44)} frames at {width}x{height_img}")
        print(f"  Time range: [{time_samples[0]:.3f}, {time_samples[-1]:.3f}]")

        # Render frames
        video_path = f"{video_dir}/traj_4d_step{step}.mp4"
        writer = imageio.get_writer(video_path, fps=cfg.render_traj_fps)

        for i in tqdm.trange(len(traj_poses_44), desc="Rendering trajectory"):
            camtoworlds = traj_poses_44[i:i+1]  # [1, 4, 4]
            Ks = K[None]  # [1, 3, 3]
            t = time_samples[i]

            # Render at this camera pose and time
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height_img,
                t=t,
                sh_degree=cfg.sh_degree,
                static_mode=False,
                use_temporal_opacity=True,
            )

            colors = torch.clamp(renders[..., :3], 0.0, 1.0)  # [1, H, W, 3]

            # Convert to uint8 and write
            frame = colors.squeeze(0).cpu().numpy()  # [H, W, 3]
            frame = (frame * 255).astype(np.uint8)
            writer.append_data(frame)

        writer.close()
        print(f"  Video saved to: {video_path}")

        # Also render duration and velocity visualization video
        self._render_traj_visualization(step, traj_poses_44, K, width, height_img, time_samples)

    @torch.no_grad()
    def _render_traj_visualization(
        self,
        step: int,
        traj_poses: Tensor,
        K: Tensor,
        width: int,
        height: int,
        time_samples: np.ndarray,
    ):
        """Render duration and velocity visualization along trajectory."""
        cfg = self.cfg
        video_dir = f"{cfg.result_dir}/videos"

        # Duration video
        dur_path = f"{video_dir}/traj_duration_step{step}.mp4"
        dur_writer = imageio.get_writer(dur_path, fps=cfg.render_traj_fps)

        # Velocity video
        vel_path = f"{video_dir}/traj_velocity_step{step}.mp4"
        vel_writer = imageio.get_writer(vel_path, fps=cfg.render_traj_fps)

        for i in tqdm.trange(len(traj_poses), desc="Rendering visualization"):
            camtoworlds = traj_poses[i:i+1]
            Ks = K[None]
            t = time_samples[i]

            dur_img, vel_img = self.render_duration_velocity_images(
                camtoworlds, Ks, width, height, t,
                static_mode=False, use_temporal_opacity=True,
                add_colorbar=False,  # No colorbar for video frames
            )

            # Duration frame
            dur_frame = dur_img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            dur_frame = (dur_frame * 255).astype(np.uint8)
            dur_writer.append_data(dur_frame)

            # Velocity frame
            vel_frame = vel_img.permute(1, 2, 0).cpu().numpy()  # [H, W, 3]
            vel_frame = (vel_frame * 255).astype(np.uint8)
            vel_writer.append_data(vel_frame)

        dur_writer.close()
        vel_writer.close()
        print(f"  Duration video saved to: {dur_path}")
        print(f"  Velocity video saved to: {vel_path}")

    @torch.no_grad()
    def export_ply_sequence(self, step: int):
        """
        Export PLY files for each frame from start_frame to end_frame.

        For each time t, exports the Gaussians with:
        - Positions computed at time t: µx(t) = µx + v·(t-µt)
        - Opacities modulated by temporal opacity: σ·σ(t)
        - Same scales, quats, and colors
        """
        print("\n[PLY Export] Exporting Gaussian sequence...")
        cfg = self.cfg
        device = self.device

        # Create export directory
        ply_dir = f"{cfg.result_dir}/ply_sequence_step{step}"
        os.makedirs(ply_dir, exist_ok=True)

        # Get frame range
        start_frame = cfg.start_frame
        end_frame = cfg.end_frame
        n_frames = end_frame - start_frame

        print(f"  Exporting {n_frames} frames from {start_frame} to {end_frame}")
        print(f"  Output directory: {ply_dir}")

        # Get base Gaussian parameters (constant across time)
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        quats = self.splats["quats"]  # [N, 4]
        sh0 = self.splats["sh0"]  # [N, 1, 3]
        shN = self.splats["shN"]  # [N, K, 3]

        for frame in tqdm.trange(n_frames, desc="Exporting PLY"):
            # Compute normalized time t in [0, 1]
            t = frame / max(n_frames - 1, 1)

            # Compute positions at time t: µx(t) = µx + v·(t-µt)
            means_t = self.compute_positions_at_time(t)  # [N, 3]

            # Compute temporal opacity: σ(t) = exp(-0.5*((t-µt)/s)^2)
            temporal_opacity = self.compute_temporal_opacity(t)  # [N]

            # Combined opacity: base_opacity * temporal_opacity
            base_opacity = torch.sigmoid(self.splats["opacities"])  # [N]
            opacities_t = base_opacity * temporal_opacity  # [N]

            # Filter out low opacity Gaussians (makes files MUCH smaller)
            # Only keep Gaussians with combined opacity > threshold
            opacity_threshold = cfg.export_ply_opacity_threshold
            valid_mask = opacities_t > opacity_threshold
            n_visible = valid_mask.sum().item()

            # Apply mask to all parameters
            means_visible = means_t[valid_mask]
            scales_visible = scales[valid_mask]
            quats_visible = quats[valid_mask]
            opacities_visible = opacities_t[valid_mask]
            sh0_visible = sh0[valid_mask]
            shN_visible = shN[valid_mask]

            if frame == 0 or frame == n_frames - 1:
                print(f"  Frame {frame}: {n_visible:,} / {len(opacities_t):,} Gaussians visible (t={t:.3f})")

            # Export to PLY (only visible Gaussians)
            filepath = os.path.join(ply_dir, f"frame_{frame:06d}.ply")
            export_splats(
                means=means_visible,
                scales=scales_visible,
                quats=quats_visible,
                opacities=opacities_visible,
                sh0=sh0_visible,
                shN=shN_visible,
                format=cfg.export_ply_format,
                save_to=filepath,
            )

        print(f"  Exported {n_frames} PLY files to {ply_dir}")


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = FreeTime4DRunner(local_rank, world_rank, world_size, cfg)

    if cfg.export_only:
        # Export mode: just generate PLY and videos from checkpoint
        runner.export_from_checkpoint()
    else:
        # Training mode (can resume from checkpoint)
        runner.train()


if __name__ == "__main__":
    """
    Usage:

    # Training with DefaultStrategy
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py default \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results \
        --start-frame 0 --end-frame 300

    # Training with MCMCStrategy
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results

    # Resume training from checkpoint
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results \
        --ckpt-path /path/to/results/ckpts/ckpt_29999.pt

    # Export PLY sequence and videos from checkpoint (no training)
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d.py mcmc \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results \
        --ckpt-path /path/to/results/ckpts/ckpt_59999.pt \
        --export-only
    """

    configs = {
        "default": (
            "FreeTimeGS 4D training with DefaultStrategy densification.",
            Config(
                strategy=DefaultStrategy(
                    verbose=True,
                    refine_start_iter=500,
                    refine_stop_iter=40_000,  # Extended from 15k to allow more densification
                    reset_every=6000,  # Less frequent resets (was 3000) to prevent mass pruning
                    refine_every=100,
                    prune_opa=0.005,
                    grow_grad2d=0.0002,
                ),
            ),
        ),
        "mcmc": (
            "FreeTimeGS 4D training with MCMCStrategy densification.",
            Config(
                init_opacity=0.5,
                init_scale=0.1,
                use_relocation=True,  # Enable BOTH: MCMC relocation + paper's periodic relocation
                relocation_every=100,  # Paper's novelty: frequent periodic relocation every 100
                strategy=MCMCStrategy(
                    verbose=True,
                    refine_start_iter=500,
                    refine_stop_iter=50_000,
                    refine_every=270,  # MCMC relocation at 270 (different from paper's 100)
                    cap_max=4_000_000,  # Max Gaussians (must be > init count from NPZ)
                ),
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
