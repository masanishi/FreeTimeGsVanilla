"""
FreeTimeGS: Gaussian Primitives at Anytime Anywhere for 4D Scene Reconstruction

Implementation based on the FreeTimeGS paper with gsplat rendering.

================================================================================
COMPLETE EXAMPLE: Training FreeTimeGS on the Elly Dataset
================================================================================

This example shows the full pipeline from raw data to trained 4D Gaussian model.

Step 1: Generate Dense Point Clouds with RoMaV2
-----------------------------------------------
First, run RoMaV2 triangulation to create dense per-frame COLMAP models.
This step takes ~30 minutes for 100 frames and creates ~600K points per frame.

    CUDA_VISIBLE_DEVICES=0 python triangulate_romav2.py \\
        --colmap_path /data/shared/elaheh/4D_demo/elly/undistorted/sparse/0 \\
        --cam_dir /data/shared/elaheh/4D_demo/elly/undistorted/images \\
        --output_dir /data/shared/elaheh/4D_demo/elly/undistorted \\
        --frame_start 0 --frame_end 100 --frame_step 1 \\
        --k_nearest 10 --num_matches 2000 --setting fast --no_compile

This creates:
    - /data/shared/.../sparse/frame_000000/  (dense COLMAP model for frame 0)
    - /data/shared/.../sparse/frame_000001/  (dense COLMAP model for frame 1)
    - ...
    - /data/shared/.../velocity_k10_step1.npz (KNN velocity estimates)

Step 2: Train FreeTimeGS
------------------------
Now train the 4D Gaussian model. The trainer automatically loads the dense
point clouds from Step 1 and initializes Gaussians with their velocities.

    # Quick test (100 steps, ~2 minutes)
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians.py default \\
        --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
        --result-dir=/data/shared/elaheh/elly_free_roma_test \\
        --start-frame=0 --end-frame=10 \\
        --max-steps=100

    # Full training (30K steps, ~2-4 hours)
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians.py default \\
        --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
        --result-dir=/data/shared/elaheh/elly_free_roma \\
        --start-frame=0 --end-frame=100 \\
        --max-steps=30000

Step 3: Monitor Training
------------------------
Open TensorBoard to see training progress:

    tensorboard --logdir=/data/shared/elaheh/elly_free_roma/tb --port=6006

Key metrics to watch:
    - train/loss: Should decrease steadily
    - train/l1_loss: Image reconstruction quality
    - train/num_gaussians: Number of active Gaussians
    - train/gt_vs_render: Side-by-side comparison images

Step 4: Results
---------------
After training, you'll find:
    - /data/shared/elaheh/elly_free_roma/ckpts/     - Model checkpoints
    - /data/shared/elaheh/elly_free_roma/renders/   - Validation images
    - /data/shared/elaheh/elly_free_roma/videos/    - Time trajectory video
    - /data/shared/elaheh/elly_free_roma/stats/     - Evaluation metrics (PSNR, SSIM, LPIPS)

================================================================================
Key Features
================================================================================
1. 4D Gaussian primitives with: position, time, duration, velocity, scale, orientation, opacity, SH
2. Motion function: µx(t) = µx + v * (t - µt)
3. Temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)^2)
4. 4D regularization loss with stop-gradient
5. Periodic relocation of low-opacity primitives
6. Velocity annealing scheduler
7. Initialization from multi-frame COLMAP with KNN velocity estimation

================================================================================
Usage Options
================================================================================

# Basic training with elly dataset
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians.py default \\
    --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
    --result-dir=/data/shared/elaheh/elly_free_roma \\
    --start-frame=0 --end-frame=100 \\
    --max-steps=30000

# Quick test run (100 steps)
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians.py fast \\
    --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
    --result-dir=/data/shared/elaheh/elly_free_roma_test \\
    --start-frame=0 --end-frame=10 \\
    --max-steps=100

# Full training with paper's approach (relocation only, no densification)
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians.py full \\
    --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
    --result-dir=/data/shared/elaheh/elly_free_roma \\
    --start-frame=0 --end-frame=100 \\
    --max-steps=30000
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

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import _update_param_with_optimizer
from gsplat.distributed import cli

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.FreeTime_dataset import (
    FreeTimeParser,
    FreeTimeDataset,
    load_multiframe_colmap_points,
    load_multiframe_colmap_grid_tracked,
    load_single_frame_with_velocity,
    load_startframe_tracked_velocity,
)
from utils import knn, rgb_to_sh, set_random_seed


@dataclass
class FreeTimeConfig:
    """Configuration for FreeTimeGS training."""

    # Data
    data_dir: str = "data/4d_scene"
    result_dir: str = "/data/shared/elaheh/elly_free_roma"
    data_factor: int = 1

    # Frame range
    start_frame: int = 0
    end_frame: int = 300
    frame_step: int = 1  # Load every N-th frame (e.g., 10 means frames 0, 10, 20, ...)

    # RoMaV2 triangulation - run before training if enabled
    run_roma: bool = False  # If True, run RoMaV2 triangulation before training
    roma_num_matches: int = 10000  # Number of RoMa matches per pair
    roma_max_pairs: int = 10  # Max camera pairs for triangulation
    roma_setting: str = "fast"  # RoMaV2 speed preset: turbo, fast, base, precise

    # Training
    max_steps: int = 30_000
    batch_size: int = 1
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])

    # Model
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    max_init_points: int = 100_000  # Fewer points to prevent color averaging from overlapping Gaussians
    init_opacity: float = 0.3  # Lower opacity to prevent saturation
    init_scale: float = 1.0  # Smaller scale to reduce overlap
    # Initial temporal duration (s in log scale: actual_duration = exp(log(s)))
    # Start with LARGE duration so ALL points are visible at ALL times initially.
    # The model will learn to reduce duration during training to specialize Gaussians.
    # With s=5.0, temporal opacity at any time t for point at mu_t:
    #   At t=0, mu_t=1.0: exp(-0.5 * (1.0/5.0)^2) = exp(-0.02) ≈ 0.98 (visible!)
    #   At t=0, mu_t=0.5: exp(-0.5 * (0.5/5.0)^2) = exp(-0.005) ≈ 0.995 (visible!)
    init_duration: float = 5.0  # Large initial duration - all Gaussians visible at all times

    # Loss weights (from paper)
    lambda_img: float = 0.8
    lambda_ssim: float = 0.2
    lambda_perc: float = 0.01
    lambda_reg: float = 0.0  # Disabled - was causing opacity collapse after canonical phase

    # Densification strategy:
    # FreeTimeGS paper uses FIXED budget N with ONLY relocation:
    #   - No split/clone/prune (DefaultStrategy DISABLED)
    #   - Only periodic relocation: move low-opacity Gaussians to high-gradient regions
    #   - This keeps N fixed throughout training
    # DefaultStrategy's pruning kills Gaussians due to temporal opacity modulation!
    use_default_strategy: bool = False  # Paper uses relocation only, no densification

    # Periodic relocation (MCMC-style, always used)
    use_periodic_relocation: bool = False  # Disabled for debugging transition issues
    relocation_every: int = 100
    relocation_start_iter: int = 1000  # Delay relocation to let training stabilize
    relocation_opacity_threshold: float = 0.02  # Moderate threshold for 4D (accounts for temporal modulation)
    lambda_grad: float = 0.5  # Weight for gradient in sampling score
    lambda_opacity: float = 0.5  # Weight for opacity in sampling score

    # Canonical phase: treat scene as static for first N iterations
    # This helps learn basic appearance before temporal dynamics
    # During canonical phase: freeze times/durations/velocities, train only spatial params
    canonical_phase_steps: int = 2000
    canonical_time: float = 0.5  # Fixed time during canonical phase (middle of sequence)

    # Transition phase: gradual 4D enablement after canonical
    # Phase 1 (0 to canonical_phase_steps): static_mode, fixed t=0.5
    # Phase 2 (canonical to canonical+transition): motion enabled, but still fixed t=0.5
    # Phase 3 (after transition): full 4D with varying time from data
    transition_phase_steps: int = 1000  # Steps to transition after canonical

    # Freeze temporal parameters - if True, don't train times/velocities at all
    # Just use initialized values and only train spatial appearance
    freeze_temporal_params: bool = True  # Recommended: keep velocities from initialization

    # Velocity annealing: λt = λ0^(1-t) + λ1^t
    # NOTE: Reduced from 1e-2 to 5e-3 for stability
    velocity_lr_start: float = 5e-3  # λ0 (reduced for stability)
    velocity_lr_end: float = 1e-4    # λ1

    # Rendering
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    antialiased: bool = False

    # Strategy - configured for 4D with careful pruning
    # Lower prune_opa since temporal modulation reduces effective opacity
    # Extend refine_stop_iter for longer training
    # IMPORTANT: Disable reset_every for 4D - opacity reset + pruning kills all Gaussians!
    strategy: DefaultStrategy = field(default_factory=lambda: DefaultStrategy(
        prune_opa=0.005,  # Moderate threshold for 4D
        grow_grad2d=0.0002,  # Default gradient threshold for densification
        grow_scale3d=0.01,  # Duplicate small Gaussians, split large ones
        refine_start_iter=500,  # Start densification after warmup
        refine_stop_iter=25_000,  # Continue densification longer for 4D
        reset_every=100_000,  # Effectively disabled - reset causes mass pruning in 4D!
        refine_every=100,  # Densify every 100 steps
        verbose=True,
    ))

    # Misc
    test_every: int = 8
    tb_every: int = 100
    tb_image_every: int = 100  # Log images to tensorboard every N steps (same as tb_every)
    disable_viewer: bool = True
    port: int = 8080

    # Initialization modes:
    # 1. "multiframe": Use points from ALL frames with their actual times
    # 2. "single_frame": Use ONE frame's points + velocity (experimental)
    # 3. "reference": Use reference COLMAP + random times
    # 4. "startframe": Use START frame points only, track through subsequent frames for velocity
    #                  All points at t=0 with computed velocity (recommended for tracking)
    # 5. "grid_tracked": Grid-based stratified sampling with temporal tracking (recommended!)
    #                    Ensures full scene coverage + better velocity from central difference
    init_mode: str = "grid_tracked"  # Default to grid_tracked for best results
    use_velocity_init: bool = True
    velocity_npz_path: Optional[str] = None  # Pre-computed velocities (for reference mode)
    knn_match_threshold: float = 0.5


def create_freetime_splats_with_optimizers(
    parser: FreeTimeParser,
    cfg: FreeTimeConfig,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Create FreeTimeGS splats with 4D parameters.

    Each Gaussian has:
    - means: [N, 3] position µx
    - times: [N, 1] temporal center µt
    - durations: [N, 1] temporal spread s (log scale)
    - velocities: [N, 3] velocity v
    - scales: [N, 3] spatial scale (log scale)
    - quats: [N, 4] orientation quaternion
    - opacities: [N] opacity σ (logit scale)
    - sh0: [N, 1, 3] DC spherical harmonics
    - shN: [N, K, 3] higher-order SH coefficients

    Two initialization modes:
    - "single_frame": Use ONE frame's points + velocity from next frame (paper's approach)
    - "multiframe": Use ALL frames' points (may cause OOM with many frames)
    - "reference": Use reference COLMAP points + KNN match for velocities
    """

    if cfg.init_mode == "single_frame":
        # Paper's approach: use ONE reference frame + velocity from matching
        print(f"[FreeTimeGS] Using single-frame initialization (paper's approach)...")

        # Get parser's transform for coordinate alignment (if normalize=True)
        parser_transform = parser.transform if hasattr(parser, 'transform') else None

        init_data = load_single_frame_with_velocity(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            reference_time=0.5,  # Middle of sequence - Gaussians move both directions
            max_error=2.0,
            match_threshold=0.1,
            transform=parser_transform,  # Apply coordinate alignment inside loader
        )

        points = init_data['positions']
        times = init_data['times']
        velocities = init_data['velocities']
        rgbs = init_data['colors']
        has_velocity = init_data['has_velocity']

        N = points.shape[0]
        print(f"[FreeTimeGS] Loaded {N} points from single frame")
        print(f"[FreeTimeGS] Points with velocity: {has_velocity.sum().item()} ({100*has_velocity.sum()/N:.1f}%)")

    elif cfg.init_mode == "multiframe":
        # Load 4D points directly from multi-frame COLMAP (accumulates all frames)
        print(f"[FreeTimeGS] Using multiframe COLMAP initialization...")

        # Get parser's transform for coordinate alignment (if normalize=True)
        parser_transform = parser.transform if hasattr(parser, 'transform') else None

        init_data = load_multiframe_colmap_points(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            frame_step=cfg.frame_step,
            max_error=1.0,  # Stricter filtering for RoMa triangulation
            match_threshold=0.1,
            transform=parser_transform,  # Apply coordinate alignment inside loader
        )

        points = init_data['positions']
        times = init_data['times']
        velocities = init_data['velocities']
        rgbs = init_data['colors']
        has_velocity = init_data['has_velocity']

        N = points.shape[0]
        print(f"[FreeTimeGS] Loaded {N} points from multi-frame COLMAP")
        print(f"[FreeTimeGS] Points with velocity: {has_velocity.sum().item()} ({100*has_velocity.sum()/N:.1f}%)")

        # Velocity statistics
        vel_norms = velocities.norm(dim=1)
        has_vel = vel_norms > 0.001
        print(f"[FreeTimeGS] Velocity stats: {has_vel.sum().item()}/{N} with velocity, "
              f"max={vel_norms.max():.4f}, mean={vel_norms[has_vel].mean():.4f}" if has_vel.any() else "no velocities")

        # Optionally limit points
        if cfg.max_init_points is not None and N > cfg.max_init_points:
            print(f"[FreeTimeGS] Sampling {cfg.max_init_points} points from {N}")
            # Prioritize points with velocity
            vel_indices = torch.where(has_velocity)[0]
            no_vel_indices = torch.where(~has_velocity)[0]

            n_vel = min(len(vel_indices), cfg.max_init_points // 2)
            n_no_vel = cfg.max_init_points - n_vel

            if n_vel < len(vel_indices):
                vel_sample = vel_indices[torch.randperm(len(vel_indices))[:n_vel]]
            else:
                vel_sample = vel_indices

            if n_no_vel < len(no_vel_indices):
                no_vel_sample = no_vel_indices[torch.randperm(len(no_vel_indices))[:n_no_vel]]
            else:
                no_vel_sample = no_vel_indices

            sample_idx = torch.cat([vel_sample, no_vel_sample])
            points = points[sample_idx]
            times = times[sample_idx]
            velocities = velocities[sample_idx]
            rgbs = rgbs[sample_idx]
            has_velocity = has_velocity[sample_idx]
            N = points.shape[0]

    elif cfg.init_mode == "startframe":
        # NEW: Load points from START frame only, track through subsequent frames for velocity
        print(f"[FreeTimeGS] Using startframe + tracked velocity initialization...")
        print(f"[FreeTimeGS] This uses ONLY frame 0 points, tracked through frames to compute velocity")

        # Get parser's transform for coordinate alignment (if normalize=True)
        parser_transform = parser.transform if hasattr(parser, 'transform') else None

        init_data = load_startframe_tracked_velocity(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            frame_step=cfg.frame_step,
            max_error=1.0,  # Stricter filtering for RoMa triangulation
            match_threshold=None,  # Use adaptive threshold based on scene scale
            transform=parser_transform,
        )

        points = init_data['positions']
        times = init_data['times']  # All zeros (t=0)
        velocities = init_data['velocities']
        rgbs = init_data['colors']
        has_velocity = init_data['has_velocity']

        N = points.shape[0]
        print(f"[FreeTimeGS] Loaded {N} points from start frame (all at t=0)")
        print(f"[FreeTimeGS] Points with velocity: {has_velocity.sum().item()} ({100*has_velocity.sum()/N:.1f}%)")

        # Add temporal diversity for points with velocity
        # Spread time centers across [0, 1] and adjust positions accordingly
        # Motion model: position(t) = position_init + velocity * (t - t_center)
        # If we set t_center = tc and position_init = position_0 + velocity * tc,
        # then at time t: position = position_0 + velocity * tc + velocity * (t - tc) = position_0 + velocity * t
        vel_mask = has_velocity
        n_vel_pts = vel_mask.sum().item()
        if n_vel_pts > 0:
            print(f"[FreeTimeGS] Adding temporal diversity to {n_vel_pts} points with velocity")
            # Assign time centers uniformly spread across [0, 1]
            vel_indices = torch.where(vel_mask)[0]
            spread_times = torch.linspace(0.0, 1.0, n_vel_pts).unsqueeze(1)
            # Shuffle to avoid spatial-temporal correlation
            perm = torch.randperm(n_vel_pts)
            spread_times = spread_times[perm]

            # Adjust positions: position_init = position_0 + velocity * t_center
            # So the motion model still works: pos(t) = pos_init + vel*(t-tc) = pos_0 + vel*t
            new_times = times.clone()
            new_points = points.clone()
            for i, idx in enumerate(vel_indices):
                tc = spread_times[i].item()
                new_times[idx] = tc
                new_points[idx] = points[idx] + velocities[idx] * tc

            times = new_times
            points = new_points
            print(f"[FreeTimeGS] Time centers spread: min={times.min():.3f}, max={times.max():.3f}")

        # Velocity statistics
        vel_norms = velocities.norm(dim=1)
        has_vel = vel_norms > 0.001
        if has_vel.any():
            print(f"[FreeTimeGS] Velocity stats: {has_vel.sum().item()}/{N} with velocity, "
                  f"max={vel_norms.max():.4f}, mean={vel_norms[has_vel].mean():.4f}")

        # Optionally limit points
        if cfg.max_init_points is not None and N > cfg.max_init_points:
            print(f"[FreeTimeGS] Sampling {cfg.max_init_points} points from {N}")
            # Prioritize points with velocity
            vel_indices = torch.where(has_velocity)[0]
            no_vel_indices = torch.where(~has_velocity)[0]

            n_vel = min(len(vel_indices), cfg.max_init_points // 2)
            n_no_vel = cfg.max_init_points - n_vel

            if n_vel < len(vel_indices):
                vel_sample = vel_indices[torch.randperm(len(vel_indices))[:n_vel]]
            else:
                vel_sample = vel_indices

            if n_no_vel < len(no_vel_indices):
                no_vel_sample = no_vel_indices[torch.randperm(len(no_vel_indices))[:n_no_vel]]
            else:
                no_vel_sample = no_vel_indices

            sample_idx = torch.cat([vel_sample, no_vel_sample])
            points = points[sample_idx]
            times = times[sample_idx]
            velocities = velocities[sample_idx]
            rgbs = rgbs[sample_idx]
            has_velocity = has_velocity[sample_idx]
            N = points.shape[0]

    elif cfg.init_mode == "grid_tracked":
        # Grid-based stratified sampling with temporal tracking
        # Best approach: ensures scene coverage + better velocity from central difference
        print(f"[FreeTimeGS] Using grid-tracked initialization (recommended)...")
        print(f"[FreeTimeGS] Grid-based stratified sampling with temporal window tracking")

        parser_transform = parser.transform if hasattr(parser, 'transform') else None

        init_data = load_multiframe_colmap_grid_tracked(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            frame_step=cfg.frame_step,
            grid_divisions=(10, 10, 4),  # 400 cells - finer grid for faster neighbor search
            max_points_per_cell=2000,
            match_threshold=0.1,
            max_error=1.0,
            transform=parser_transform,
        )

        points = init_data['positions']
        times = init_data['times']
        velocities = init_data['velocities']
        rgbs = init_data['colors']
        has_velocity = init_data['has_velocity']

        N = points.shape[0]
        print(f"[FreeTimeGS] Loaded {N} points with grid-tracked initialization")
        print(f"[FreeTimeGS] Points with velocity: {has_velocity.sum().item()} ({100*has_velocity.float().mean():.1f}%)")

        # Velocity statistics
        vel_norms = velocities.norm(dim=1)
        has_vel = vel_norms > 0.001
        if has_vel.any():
            print(f"[FreeTimeGS] Velocity stats: max={vel_norms.max():.4f}, mean={vel_norms[has_vel].mean():.4f}")

    else:
        # Original mode: use reference COLMAP
        print(f"[FreeTimeGS] Using reference COLMAP initialization...")
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        N = points.shape[0]

        # Time: initialize uniformly in [0, 1]
        times = torch.rand((N, 1))

        # Velocity: initialize to zero
        velocities = torch.zeros((N, 3))

        print(f"[FreeTimeGS] Initializing {N} Gaussians from reference COLMAP")

        # Load pre-computed velocities if available
        if cfg.use_velocity_init and cfg.velocity_npz_path and os.path.exists(cfg.velocity_npz_path):
            print(f"[FreeTimeGS] Loading velocities from {cfg.velocity_npz_path}")
            vel_data = np.load(cfg.velocity_npz_path)

            vel_positions = torch.from_numpy(vel_data['initial_positions']).float()
            vel_velocities = torch.from_numpy(vel_data['velocities']).float()

            # Match points to velocity bank using KNN
            from scipy.spatial import cKDTree
            tree = cKDTree(vel_positions.numpy())
            distances, indices = tree.query(points.numpy(), k=1)

            valid_matches = distances < cfg.knn_match_threshold
            velocities[valid_matches] = vel_velocities[indices[valid_matches]]

            n_matched = valid_matches.sum()
            print(f"[FreeTimeGS] Matched {n_matched}/{N} points with velocities")

    # Filter points that are too far from camera center
    # After normalization, cameras are at distance ~1 from origin
    # Keep points within a reasonable multiple of scene_scale
    scene_scale = parser.scene_scale
    max_point_dist = 5.0 * scene_scale  # Keep points within 5x scene scale
    point_dists = torch.norm(points, dim=1)
    valid_points = point_dists < max_point_dist

    n_before = len(points)
    points = points[valid_points]
    rgbs = rgbs[valid_points]
    times = times[valid_points]
    velocities = velocities[valid_points]
    if 'has_velocity' in dir() and has_velocity is not None:
        has_velocity = has_velocity[valid_points]

    n_after = len(points)
    print(f"[FreeTimeGS] Filtered points by distance: {n_before} -> {n_after} "
          f"(removed {n_before - n_after} points beyond {max_point_dist:.1f} units)")

    # Subsample if too many points (prevents color averaging from overlapping Gaussians)
    max_pts = cfg.max_init_points if cfg.max_init_points is not None else 500_000
    if max_pts > 0 and len(points) > max_pts:
        idx = torch.randperm(len(points))[:max_pts]
        points = points[idx]
        rgbs = rgbs[idx]
        times = times[idx]
        velocities = velocities[idx]
        if 'has_velocity' in dir() and has_velocity is not None:
            has_velocity = has_velocity[idx]
        print(f"[FreeTimeGS] Subsampled to {max_pts} points (was {n_after})")

    N = points.shape[0]

    # Initialize spatial scale from KNN
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    # Clamp to avoid log(0) = -inf for duplicate or very close points
    dist_avg = torch.clamp(dist_avg, min=1e-6)
    scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

    # Distribute across ranks
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    times = times[world_rank::world_size]
    velocities = velocities[world_rank::world_size]
    N = points.shape[0]

    # Initialize other 4D parameters
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), cfg.init_opacity))

    # Duration: initialize to cfg.init_duration (log scale for positivity)
    durations = torch.log(torch.full((N, 1), cfg.init_duration))

    # Spherical harmonics
    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    # Build parameter list with learning rates (scene_scale defined above)
    # NOTE: Learning rates balance stability vs convergence speed
    # - Higher LR for scales/opacities to allow Gaussians to grow and become visible
    # - Lower LR for temporal params (times, durations, velocities) for stability
    params = [
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("times", torch.nn.Parameter(times), 5e-4),      # Low: time should change slowly
        ("durations", torch.nn.Parameter(durations), 5e-4),  # Low: duration should change slowly
        ("velocities", torch.nn.Parameter(velocities), cfg.velocity_lr_start),  # 5e-3, annealed
        ("scales", torch.nn.Parameter(scales), 1e-2),    # High LR to let scales grow quickly
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 2e-2),  # Higher to let opacity increase
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # DEBUG: Print initialization statistics
    print("\n" + "="*70)
    print("[INIT DEBUG] Gaussian initialization statistics:")
    print("="*70)
    print(f"  Number of Gaussians: {N}")
    print(f"  Positions: min={points.min():.4f}, max={points.max():.4f}, mean={points.mean():.4f}")
    print(f"  RGB colors: min={rgbs.min():.4f}, max={rgbs.max():.4f}, mean={rgbs.mean():.4f}")
    sh0_vals = colors[:, 0, :]
    print(f"  SH0 (DC): min={sh0_vals.min():.4f}, max={sh0_vals.max():.4f}, mean={sh0_vals.mean():.4f}")
    scales_actual = torch.exp(scales)
    print(f"  Scales (stored log): min={scales.min():.4f}, max={scales.max():.4f}")
    print(f"  Scales (actual): min={scales_actual.min():.4f}, max={scales_actual.max():.4f}, mean={scales_actual.mean():.4f}")
    opacities_actual = torch.sigmoid(opacities)
    print(f"  Opacities (stored logit): {opacities[0].item():.4f}")
    print(f"  Opacities (actual sigmoid): {opacities_actual[0].item():.4f}")
    print(f"  Durations (stored log): {durations[0].item():.4f}")
    print(f"  Durations (actual exp): {torch.exp(durations[0]).item():.4f}")
    print(f"  Times: min={times.min():.4f}, max={times.max():.4f}, mean={times.mean():.4f}")
    print(f"  Velocities: min={velocities.min():.4f}, max={velocities.max():.4f}, mean={velocities.mean():.4f}")
    print(f"  Scene scale: {scene_scale:.4f}")
    print("="*70 + "\n")

    # Create optimizers
    BS = cfg.batch_size * world_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers


class FreeTimeGSRunner:
    """Training runner for FreeTimeGS."""

    def __init__(
        self,
        local_rank: int,
        world_rank: int,
        world_size: int,
        cfg: FreeTimeConfig,
    ):
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

        # Load data
        self.parser = FreeTimeParser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=True,
            test_every=cfg.test_every,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
        )

        self.trainset = FreeTimeDataset(self.parser, split="train")
        self.valset = FreeTimeDataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1

        print(f"[FreeTimeGS] Scene scale: {self.scene_scale}")
        print(f"[FreeTimeGS] Train samples: {len(self.trainset)}")
        print(f"[FreeTimeGS] Val samples: {len(self.valset)}")

        # Create model
        self.splats, self.optimizers = create_freetime_splats_with_optimizers(
            self.parser,
            cfg,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )

        print(f"[FreeTimeGS] Initialized {len(self.splats['means'])} Gaussians")

        # Strategy for densification
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)

        # Losses
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

    def compute_temporal_opacity(self, t: float) -> Tensor:
        """
        Compute temporal opacity σ(t) for all Gaussians at time t.

        σ(t) = exp(-0.5 * ((t - µt) / s)^2)

        Args:
            t: Query time in [0, 1]

        Returns:
            [N] tensor of temporal opacities
        """
        mu_t = self.splats["times"]  # [N, 1]
        s = torch.exp(self.splats["durations"])  # [N, 1] - duration in positive space

        # Temporal Gaussian
        temporal_opacity = torch.exp(-0.5 * ((t - mu_t) / (s + 1e-8)) ** 2)

        return temporal_opacity.squeeze(-1)  # [N]

    def compute_moved_positions(self, t: float) -> Tensor:
        """
        Compute moved positions µx(t) = µx + v * (t - µt)

        Args:
            t: Query time in [0, 1]

        Returns:
            [N, 3] tensor of moved positions
        """
        mu_x = self.splats["means"]  # [N, 3]
        mu_t = self.splats["times"]  # [N, 1]
        v = self.splats["velocities"]  # [N, 3]

        # Motion function
        dt = t - mu_t  # [N, 1]
        moved_positions = mu_x + v * dt  # [N, 3]

        return moved_positions

    def rasterize_at_time(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        t: float,
        sh_degree: int,
        static_mode: bool = False,  # Canonical phase: no motion, all visible
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize Gaussians at a specific time t.

        From FreeTimeGS paper:
        1. Compute moved positions: µx(t) = µx + v * (t - µt)  (Eq. 1)
        2. Compute temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)^2)  (Eq. 4)
        3. Modulate opacity: σ_effective = σ * σ(t)  (Eq. 3)

        If static_mode=True (canonical phase):
        - Use original positions (no motion)
        - Set temporal_opacity = 1.0 (all Gaussians visible)
        """

        if static_mode:
            # CANONICAL PHASE: static scene, all Gaussians visible
            means = self.splats["means"]  # Original positions, no motion
            temporal_opacity = torch.ones(len(means), device=self.device)  # All visible
        else:
            # 4D mode: apply motion and temporal opacity
            means = self.compute_moved_positions(t)  # [N, 3]
            temporal_opacity = self.compute_temporal_opacity(t)  # [N]

        # Standard Gaussian parameters
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        base_opacity = torch.sigmoid(self.splats["opacities"])  # [N]

        # Modulate opacity with temporal opacity (Eq. 3)
        opacities = base_opacity * temporal_opacity  # [N]

        # Spherical harmonics
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]
        colors = torch.cat([sh0, shN], dim=1)  # [N, K, 3]

        # Rasterize
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"

        # Extract near/far plane from kwargs or use config defaults
        near_plane = kwargs.pop("near_plane", self.cfg.near_plane)
        far_plane = kwargs.pop("far_plane", self.cfg.far_plane)

        render_colors, render_alphas, info = rasterization(
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
            absgrad=self.cfg.strategy.absgrad if hasattr(self.cfg.strategy, 'absgrad') else False,
            rasterize_mode=rasterize_mode,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            **kwargs,
        )

        # Store temporal opacity for regularization
        info["temporal_opacity"] = temporal_opacity

        return render_colors, render_alphas, info

    def compute_4d_regularization(self, temporal_opacity: Tensor) -> Tensor:
        """
        Compute 4D regularization loss.

        L_reg(t) = (1/N) * Σ(σ * sg[σ(t)])

        The stop-gradient on σ(t) prevents the regularization from
        minimizing temporal opacity.
        """
        base_opacity = torch.sigmoid(self.splats["opacities"])  # [N]

        # Stop gradient on temporal opacity
        temporal_opacity_sg = temporal_opacity.detach()

        # Weighted regularization
        reg_loss = (base_opacity * temporal_opacity_sg).mean()

        return reg_loss

    def get_velocity_lr_scale(self, step: int, max_steps: int) -> float:
        """
        Compute velocity learning rate scale using annealing.

        λt = λ0^(1-t) + λ1^t, where t ∈ [0, 1]
        """
        t = step / max_steps
        lambda_0 = self.cfg.velocity_lr_start
        lambda_1 = self.cfg.velocity_lr_end

        # Annealing formula from paper: λt = λ0^(1-t) + λ1^t
        # Uses addition not multiplication for proper interpolation
        lr_scale = (lambda_0 ** (1 - t)) + (lambda_1 ** t)

        return lr_scale / lambda_0  # Return relative scale

    def periodic_relocation(self, step: int, info: Dict):
        """
        Relocate low-opacity Gaussians to high-score regions.

        This is similar to MCMC strategy's relocation but uses a sampling score
        based on both gradient and opacity: s = λg * ∇g + λo * σ

        Without periodic relocation, the model tends to use more Gaussians
        with lower opacity to model the scene, leading to suboptimal results
        when the number of Gaussians is limited (as noted in FreeTimeGS paper).

        The relocation properly updates optimizer states for all parameters
        including 4D parameters (times, durations, velocities).
        """
        with torch.no_grad():
            base_opacity = torch.sigmoid(self.splats["opacities"])

            # Get spatial gradients if available (accumulated from strategy state)
            if self.strategy_state.get("grad2d") is not None:
                count = self.strategy_state["count"].clamp_min(1)
                spatial_grad = self.strategy_state["grad2d"] / count
            elif "means2d" in info and self.splats["means"].grad is not None:
                spatial_grad = self.splats["means"].grad.norm(dim=-1)
            else:
                spatial_grad = torch.zeros_like(base_opacity)

            # Compute sampling score: s = λg * ∇g + λo * σ
            sampling_score = (
                self.cfg.lambda_grad * spatial_grad +
                self.cfg.lambda_opacity * base_opacity
            )

            # Find low-opacity (dead) Gaussians to relocate
            dead_mask = base_opacity < self.cfg.relocation_opacity_threshold
            n_dead = dead_mask.sum().item()

            if n_dead == 0:
                return 0

            # Find alive Gaussians to sample from
            alive_mask = ~dead_mask
            if alive_mask.sum() == 0:
                return 0

            dead_indices = dead_mask.nonzero(as_tuple=True)[0]
            alive_indices = alive_mask.nonzero(as_tuple=True)[0]

            # Probability proportional to sampling score of alive Gaussians
            probs = sampling_score[alive_indices]
            probs = probs / probs.sum().clamp_min(1e-8)

            # Sample source indices from alive Gaussians
            sampled_idxs = torch.multinomial(probs, n_dead, replacement=True)
            source_indices = alive_indices[sampled_idxs]

            # Relocate: copy all parameters from source to dead Gaussians
            # This properly handles all 4D parameters (times, durations, velocities)
            def param_fn(name: str, p: Tensor) -> Tensor:
                if name == "means":
                    # Add small noise to positions
                    noise_scale = 0.01 * self.scene_scale
                    noise = torch.randn(n_dead, 3, device=self.device) * noise_scale
                    p[dead_indices] = p[source_indices] + noise
                elif name == "opacities":
                    # Reset opacity to initial value
                    p[dead_indices] = torch.logit(
                        torch.full((n_dead,), self.cfg.init_opacity, device=self.device)
                    )
                else:
                    # Copy other parameters (scales, quats, sh0, shN, times, durations, velocities)
                    p[dead_indices] = p[source_indices]
                return torch.nn.Parameter(p, requires_grad=p.requires_grad)

            def optimizer_fn(key: str, v: Tensor) -> Tensor:
                # Reset optimizer state for relocated Gaussians
                v[dead_indices] = 0
                return v

            # Update parameters and optimizer states
            _update_param_with_optimizer(
                param_fn, optimizer_fn, self.splats, self.optimizers
            )

            # Update strategy state if it exists
            for k, v in self.strategy_state.items():
                if isinstance(v, torch.Tensor) and v.shape[0] == len(base_opacity):
                    v[dead_indices] = 0

            print(f"[Relocation] Relocated {n_dead} low-opacity Gaussians at step {step}")
            return n_dead

    def train(self):
        cfg = self.cfg
        device = self.device
        max_steps = cfg.max_steps

        # Dump config
        if self.world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        # Learning rate schedulers
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]

        # Data loader
        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop
        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))

        # Log canonical phase settings
        if cfg.canonical_phase_steps > 0:
            print(f"\n[CANONICAL PHASE] First {cfg.canonical_phase_steps} steps: static scene at t={cfg.canonical_time}")
            print(f"  Frozen: times, durations, velocities")
            print(f"  Training: means, scales, quats, opacities, colors")
            print(f"  After step {cfg.canonical_phase_steps}: full 4D training\n")

        for step in pbar:
            # Load batch
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)  # [B, 4, 4]
            Ks = data["K"].to(device)  # [B, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [B, H, W, 3]

            # THREE-PHASE TRAINING:
            # Phase 1 (canonical): static_mode, t=0.5, temporal params frozen
            # Phase 2 (transition): motion enabled, t=0.5 still fixed, temporal params learning
            # Phase 3 (full 4D): motion enabled, t from data
            transition_end = cfg.canonical_phase_steps + cfg.transition_phase_steps
            in_canonical_phase = step < cfg.canonical_phase_steps
            in_transition_phase = cfg.canonical_phase_steps <= step < transition_end
            in_full_4d_phase = step >= transition_end

            if in_canonical_phase or in_transition_phase:
                t = cfg.canonical_time  # Fixed time (default 0.5 = middle)
            else:
                t = data["time"].to(device).mean().item()  # Actual time from data

            # IMPORTANT: Don't use static_mode anymore!
            # static_mode causes position discontinuity when transitioning to 4D
            # Instead, always use normal 4D rendering (motion applied)
            # During canonical phase, temporal params are frozen but motion is still applied
            # This ensures positions are consistent: pos = means + vel * (t - mu_t)
            use_static_mode = False  # Always use 4D rendering for consistent positions

            # Log phase transitions
            if step == cfg.canonical_phase_steps:
                print(f"\n" + "="*70)
                print(f"[TRANSITION PHASE START] Step {step}")
                print(f"  Motion ENABLED, but still fixed t={cfg.canonical_time}")
                print(f"  Now training: times, durations, velocities")
                print(f"  Transition ends at step {transition_end}")
                print(f"="*70)

            if step == transition_end:
                print(f"\n" + "="*70)
                print(f"[FULL 4D PHASE START] Step {step}")
                print(f"  Now using varying time t from data")
                print(f"="*70)
                # Debug: show current state of 4D parameters
                with torch.no_grad():
                    times = self.splats["times"].squeeze()
                    durations = torch.exp(self.splats["durations"]).squeeze()
                    velocities = self.splats["velocities"]
                    means = self.splats["means"]
                    base_op = torch.sigmoid(self.splats["opacities"])
                    print(f"  Times: min={times.min():.4f}, max={times.max():.4f}, mean={times.mean():.4f}")
                    print(f"  Durations: min={durations.min():.4f}, max={durations.max():.4f}, mean={durations.mean():.4f}")
                    print(f"  Velocities: min={velocities.min():.4f}, max={velocities.max():.4f}, mean={velocities.mean():.4f}")
                    print(f"  Velocity magnitudes: min={velocities.norm(dim=-1).min():.4f}, max={velocities.norm(dim=-1).max():.4f}")
                    print(f"  Base opacity: min={base_op.min():.4f}, max={base_op.max():.4f}, mean={base_op.mean():.4f}")
                    print(f"  Means: min={means.min():.4f}, max={means.max():.4f}")
                    # Test: what would temporal opacity be at t=0 vs t=1?
                    for test_t in [0.0, 0.5, 1.0]:
                        dt = test_t - times
                        temp_op = torch.exp(-0.5 * (dt / (durations + 1e-8)) ** 2)
                        # Also compute position displacement
                        displacement = velocities * dt.unsqueeze(-1)
                        disp_mag = displacement.norm(dim=-1)
                        print(f"  At t={test_t}: temp_op mean={temp_op.mean():.4f}, position displacement: min={disp_mag.min():.4f}, max={disp_mag.max():.4f}, mean={disp_mag.mean():.4f}")
                print(f"="*70 + "\n")

            height, width = pixels.shape[1:3]

            # SH degree schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Update temporal parameter learning rates
            # If freeze_temporal_params=True, keep LR=0 always (use initialized values)
            if cfg.freeze_temporal_params or in_canonical_phase:
                vel_lr = 0.0
                times_lr = 0.0
                durations_lr = 0.0
            else:
                vel_lr_scale = self.get_velocity_lr_scale(step, max_steps)
                vel_lr = cfg.velocity_lr_start * vel_lr_scale * math.sqrt(cfg.batch_size)
                times_lr = 5e-4 * math.sqrt(cfg.batch_size)
                durations_lr = 5e-4 * math.sqrt(cfg.batch_size)

            for param_group in self.optimizers["velocities"].param_groups:
                param_group["lr"] = vel_lr
            for param_group in self.optimizers["times"].param_groups:
                param_group["lr"] = times_lr
            for param_group in self.optimizers["durations"].param_groups:
                param_group["lr"] = durations_lr

            # Forward pass: render at time t
            renders, alphas, info = self.rasterize_at_time(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                t=t,
                sh_degree=sh_degree_to_use,
                static_mode=use_static_mode,  # Only static during canonical, transition uses motion
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )

            colors = renders[..., :3]

            # DEBUG: Log rendering statistics every 100 steps
            if step % 100 == 0:
                temporal_op = info["temporal_opacity"]
                base_op = torch.sigmoid(self.splats["opacities"])
                effective_op = base_op * temporal_op

                # Get Gaussian times
                mu_t = self.splats["times"].squeeze()

                if in_canonical_phase:
                    phase_label = "CANONICAL/STATIC"
                elif in_transition_phase:
                    phase_label = "TRANSITION (t=0.5, motion ON)"
                else:
                    phase_label = "FULL 4D"
                print(f"\n[DEBUG step={step}] t={t:.3f} mode={phase_label}")
                print(f"  Gaussian times: min={mu_t.min():.3f}, max={mu_t.max():.3f}, mean={mu_t.mean():.3f}")
                print(f"  Temporal opacity: min={temporal_op.min():.4f}, max={temporal_op.max():.4f}, mean={temporal_op.mean():.4f}")
                print(f"  Base opacity: min={base_op.min():.4f}, max={base_op.max():.4f}, mean={base_op.mean():.4f}")
                print(f"  Effective opacity: min={effective_op.min():.4f}, max={effective_op.max():.4f}, mean={effective_op.mean():.4f}")
                print(f"  Render alpha: min={alphas.min():.4f}, max={alphas.max():.4f}, mean={alphas.mean():.4f}")
                print(f"  Render colors: min={colors.min():.4f}, max={colors.max():.4f}, mean={colors.mean():.4f}")
                print(f"  GT pixels: min={pixels.min():.4f}, max={pixels.max():.4f}, mean={pixels.mean():.4f}")

                # DEBUG: Check positions and scales
                means = self.splats["means"]
                scales = torch.exp(self.splats["scales"])
                cam_pos = camtoworlds[0, :3, 3]
                print(f"  Gaussian means: min={means.min():.3f}, max={means.max():.3f}")
                print(f"  Gaussian scales: min={scales.min():.4f}, max={scales.max():.4f}, mean={scales.mean():.4f}")
                print(f"  Camera position: {cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}")

            # Strategy pre-backward (only if using DefaultStrategy)
            if cfg.use_default_strategy:
                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

            # Compute losses
            # Clamp colors to [0, 1] for loss computation (SH can produce values > 1)
            colors_clamped = torch.clamp(colors, 0.0, 1.0)

            # Image loss (L1)
            l1_loss = F.l1_loss(colors_clamped, pixels)

            # SSIM loss
            ssim_loss = 1.0 - fused_ssim(
                colors_clamped.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid"
            )

            # Perceptual loss (LPIPS) - requires [0, 1] range
            lpips_loss = self.lpips(
                colors_clamped.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2)
            )

            # Combined rendering loss
            render_loss = (
                cfg.lambda_img * l1_loss +
                cfg.lambda_ssim * ssim_loss +
                cfg.lambda_perc * lpips_loss
            )

            # 4D regularization loss (disabled during canonical phase to prevent opacity collapse)
            if in_canonical_phase:
                reg_loss = torch.zeros(1, device=self.device)
            else:
                reg_loss = self.compute_4d_regularization(info["temporal_opacity"])

            # Total loss
            loss = render_loss + cfg.lambda_reg * reg_loss

            # Backward
            loss.backward()

            # Progress bar
            if in_canonical_phase:
                phase_str = "[STATIC]"
            elif in_transition_phase:
                phase_str = "[TRANS]"
            else:
                phase_str = "[4D]"
            desc = (
                f"{phase_str} loss={loss.item():.4f} | "
                f"l1={l1_loss.item():.4f} | "
                f"reg={reg_loss.item():.4f} | "
                f"t={t:.2f} | "
                f"N={len(self.splats['means'])}"
            )
            pbar.set_description(desc)

            # Tensorboard logging
            if self.world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1_loss", l1_loss.item(), step)
                self.writer.add_scalar("train/ssim_loss", ssim_loss.item(), step)
                self.writer.add_scalar("train/lpips_loss", lpips_loss.item(), step)
                self.writer.add_scalar("train/reg_loss", reg_loss.item(), step)
                self.writer.add_scalar("train/num_gaussians", len(self.splats["means"]), step)
                self.writer.add_scalar("train/vel_lr", vel_lr, step)
                self.writer.add_scalar("train/in_canonical_phase", float(in_canonical_phase), step)

                # Velocity statistics (guard against empty)
                if len(self.splats["velocities"]) > 0:
                    vel_mag = self.splats["velocities"].detach().norm(dim=-1)
                    self.writer.add_scalar("train/vel_mean", vel_mag.mean().item(), step)
                    self.writer.add_scalar("train/vel_max", vel_mag.max().item(), step)

                # Temporal statistics (guard against empty)
                if len(self.splats["durations"]) > 0:
                    durations = torch.exp(self.splats["durations"]).detach()
                    self.writer.add_scalar("train/duration_mean", durations.mean().item(), step)
                    self.writer.add_scalar("train/duration_min", durations.min().item(), step)
                    self.writer.add_scalar("train/duration_max", durations.max().item(), step)

                # Opacity statistics
                if len(self.splats["opacities"]) > 0:
                    base_opacity = torch.sigmoid(self.splats["opacities"]).detach()
                    temporal_opacity = info["temporal_opacity"].detach()
                    effective_opacity = base_opacity * temporal_opacity

                    self.writer.add_scalar("train/base_opacity_mean", base_opacity.mean().item(), step)
                    self.writer.add_scalar("train/temporal_opacity_mean", temporal_opacity.mean().item(), step)
                    self.writer.add_scalar("train/effective_opacity_mean", effective_opacity.mean().item(), step)

                    # Count visible Gaussians (effective opacity > threshold)
                    n_visible = (effective_opacity > 0.01).sum().item()
                    self.writer.add_scalar("train/n_visible_gaussians", n_visible, step)

                # Time statistics
                if len(self.splats["times"]) > 0:
                    times_param = self.splats["times"].detach()
                    self.writer.add_scalar("train/time_mean", times_param.mean().item(), step)
                    self.writer.add_scalar("train/time_std", times_param.std().item(), step)

                # Alpha (coverage) from rendered image
                self.writer.add_scalar("train/alpha_mean", alphas.mean().item(), step)
                self.writer.add_scalar("train/alpha_max", alphas.max().item(), step)

                # Scale statistics (important for detecting collapse)
                if len(self.splats["scales"]) > 0:
                    scales_actual = torch.exp(self.splats["scales"]).detach()
                    self.writer.add_scalar("train/scale_mean", scales_actual.mean().item(), step)
                    self.writer.add_scalar("train/scale_min", scales_actual.min().item(), step)
                    self.writer.add_scalar("train/scale_max", scales_actual.max().item(), step)

                    # Warn if scales are getting too small
                    if scales_actual.mean().item() < 1e-4:
                        print(f"[WARNING] Scales shrinking! mean={scales_actual.mean():.6f}")

                # Position statistics (detect drift)
                if len(self.splats["means"]) > 0:
                    means = self.splats["means"].detach()
                    self.writer.add_scalar("train/position_std", means.std().item(), step)

                # Log images: rendered vs ground truth vs alpha (side by side)
                if step % cfg.tb_image_every == 0:
                    rendered = colors_clamped[0].detach()  # [H, W, 3] - already clamped
                    gt = pixels[0].detach()  # [H, W, 3]
                    alpha = alphas[0].detach()  # [H, W, 1]

                    # Convert alpha to RGB (grayscale)
                    alpha_rgb = alpha.expand(-1, -1, 3)  # [H, W, 3]

                    # Create side-by-side comparison: [GT | Rendered]
                    comparison = torch.cat([gt, rendered], dim=1)  # [H, 2W, 3]

                    # Log GT | Rendered comparison
                    self.writer.add_image(
                        "train/gt_render",
                        comparison.permute(2, 0, 1).contiguous(),
                        step,
                    )

                    # Log alpha separately
                    self.writer.add_image(
                        "train/alpha",
                        alpha_rgb.permute(2, 0, 1).contiguous(),
                        step,
                    )

                    print(f"[TensorBoard] Logged images at step {step}, t={t:.2f}, alpha_mean={alpha.mean():.3f}")

                self.writer.flush()

            # Save checkpoints
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                self._save_checkpoint(step)

            # Optimizer step
            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            for scheduler in schedulers:
                scheduler.step()

            # Strategy post-backward (densification) - only if enabled
            # FreeTimeGS paper uses fixed N with relocation only (no split/clone/prune)
            if cfg.use_default_strategy and isinstance(self.cfg.strategy, DefaultStrategy):
                n_before = len(self.splats["means"])
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
                n_after = len(self.splats["means"])
                if n_after != n_before:
                    print(f"[Densification] Step {step}: {n_before} -> {n_after} Gaussians "
                          f"(+{n_after - n_before})" if n_after > n_before else
                          f"[Densification] Step {step}: {n_before} -> {n_after} Gaussians "
                          f"({n_after - n_before})")
                    if self.world_rank == 0:
                        self.writer.add_scalar("train/num_gaussians", n_after, step)

            # Safety check: stop if all Gaussians are pruned
            if len(self.splats["means"]) == 0:
                print(f"[FreeTimeGS] ERROR: All Gaussians pruned at step {step}!")
                print("  This usually means:")
                print("  1. init_duration is too small (Gaussians invisible at most times)")
                print("  2. Temporal opacity too low causing aggressive pruning")
                print("  Try increasing --init_duration (e.g., 0.5) or disable pruning initially")
                break

            # Periodic relocation (FreeTimeGS-specific)
            # This works alongside DefaultStrategy densification:
            # - DefaultStrategy: split/clone/prune based on gradients and opacity
            # - Periodic relocation: move low-opacity Gaussians to high-score regions
            # Without this, model uses more low-opacity Gaussians (suboptimal when N is limited)
            # Disabled during canonical phase since we're training static scene
            if cfg.use_periodic_relocation and not in_canonical_phase and step >= cfg.relocation_start_iter and step % cfg.relocation_every == 0:
                n_relocated = self.periodic_relocation(step, info)
                if self.world_rank == 0:
                    self.writer.add_scalar("train/n_relocated", n_relocated, step)

            # Evaluation
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)

        print(f"[FreeTimeGS] Training completed in {time.time() - global_tic:.1f}s")

    def _save_checkpoint(self, step: int):
        """Save model checkpoint."""
        data = {
            "step": step,
            "splats": self.splats.state_dict(),
        }
        torch.save(data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")
        print(f"[FreeTimeGS] Saved checkpoint at step {step}")

    @torch.no_grad()
    def eval(self, step: int):
        """Evaluate on validation set."""
        print(f"[FreeTimeGS] Running evaluation at step {step}...")

        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        metrics = defaultdict(list)
        render_times = []

        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            t = data["time"].to(device).mean().item()
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            colors, _, _ = self.rasterize_at_time(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                t=t,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )

            torch.cuda.synchronize()
            render_times.append(time.time() - tic)

            colors = torch.clamp(colors[..., :3], 0.0, 1.0)

            # Save rendered images
            if self.world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(f"{self.render_dir}/val_step{step}_{i:04d}.png", canvas)

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

        if self.world_rank == 0:
            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats["render_time"] = np.mean(render_times)
            stats["num_gaussians"] = len(self.splats["means"])

            print(
                f"[Eval] PSNR: {stats['psnr']:.3f}, "
                f"SSIM: {stats['ssim']:.4f}, "
                f"LPIPS: {stats['lpips']:.4f}, "
                f"Time: {stats['render_time']:.3f}s"
            )

            with open(f"{self.stats_dir}/val_step{step}.json", "w") as f:
                json.dump(stats, f)

            for k, v in stats.items():
                self.writer.add_scalar(f"val/{k}", v, step)
            self.writer.flush()

    @torch.no_grad()
    def render_trajectory(self, step: int, n_frames: int = 100):
        """Render a trajectory video across time."""
        print(f"[FreeTimeGS] Rendering trajectory at step {step}...")

        cfg = self.cfg
        device = self.device

        # Use first camera pose
        data = self.valset[0]
        camtoworld = data["camtoworld"].unsqueeze(0).to(device)
        K = data["K"].unsqueeze(0).to(device)
        height, width = data["image"].shape[:2]
        height = height // cfg.data_factor
        width = width // cfg.data_factor

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(f"{video_dir}/trajectory_step{step}.mp4", fps=30)

        for i in tqdm.trange(n_frames, desc="Rendering trajectory"):
            t = i / (n_frames - 1)  # Time from 0 to 1

            colors, _, _ = self.rasterize_at_time(
                camtoworlds=camtoworld,
                Ks=K,
                width=width,
                height=height,
                t=t,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )

            colors = torch.clamp(colors[..., :3], 0.0, 1.0)
            frame = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(frame)

        writer.close()
        print(f"[FreeTimeGS] Video saved to {video_dir}/trajectory_step{step}.mp4")


def run_roma_triangulation(cfg: FreeTimeConfig):
    """Run RoMaV2 triangulation before training if --run_roma is enabled."""
    import subprocess

    print("\n" + "=" * 80)
    print("[RoMaV2] Running triangulation before training...")
    print("=" * 80)

    # Build the triangulate command
    cmd = [
        "python", "triangulate_romav2.py",
        "--cam_dir", os.path.join(cfg.data_dir, "sparse/0"),
        "--frame_start", str(cfg.start_frame),
        "--frame_end", str(cfg.end_frame),
        "--frame_step", str(cfg.frame_step),
        "--num_matches", str(cfg.roma_num_matches),
        "--max_pairs", str(cfg.roma_max_pairs),
        "--setting", cfg.roma_setting,
        "--output_dir", os.path.join(cfg.data_dir, "sparse"),
    ]

    print(f"[RoMaV2] Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("[RoMaV2] Triangulation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"[RoMaV2] WARNING: Triangulation failed with exit code {e.returncode}")
        print("[RoMaV2] Continuing with existing sparse models...")
    except FileNotFoundError:
        print("[RoMaV2] WARNING: triangulate_romav2.py not found")
        print("[RoMaV2] Continuing with existing sparse models...")

    print("=" * 80 + "\n")


def main(local_rank: int, world_rank: int, world_size: int, cfg: FreeTimeConfig):
    """Main entry point."""

    # Run RoMaV2 triangulation if enabled (only on rank 0)
    if cfg.run_roma and local_rank == 0:
        run_roma_triangulation(cfg)

    runner = FreeTimeGSRunner(local_rank, world_rank, world_size, cfg)
    runner.train()

    # Render final trajectory
    runner.render_trajectory(step=cfg.max_steps - 1)


if __name__ == "__main__":
    import argparse

    # Check for --single-gpu flag before tyro parsing
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--single-gpu", action="store_true",
                           help="Run on single GPU (no distributed training)")
    pre_args, remaining = pre_parser.parse_known_args()

    # Remove --single-gpu from sys.argv so tyro doesn't see it
    if pre_args.single_gpu:
        sys.argv = [sys.argv[0]] + remaining

    configs = {
        "default": (
            "FreeTimeGS training with default settings.",
            FreeTimeConfig(),
        ),
        "fast": (
            "Fast training with lower quality for testing.",
            FreeTimeConfig(
                max_steps=7_000,
                eval_steps=[7_000],
                save_steps=[7_000],
                data_factor=4,
                max_init_points=100_000,  # Limit points to avoid OOM
            ),
        ),
        "full": (
            "Full training with relocation only (paper's approach).",
            FreeTimeConfig(
                max_steps=30_000,
                eval_steps=[7_000, 15_000, 30_000],
                save_steps=[7_000, 15_000, 30_000],
                data_factor=2,  # Higher resolution (factor=1 for full res)
                max_init_points=200_000,  # More initial points
                use_default_strategy=False,  # Paper uses relocation only, NO densification/pruning
                use_periodic_relocation=True,
                relocation_every=100,  # Paper: every 100 iterations
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)

    if pre_args.single_gpu:
        # Single GPU mode: run directly without distributed wrapper
        print("[FreeTimeGS] Running in single-GPU mode")
        main(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    else:
        # Multi-GPU distributed mode
        cli(main, cfg, verbose=True)
