"""
FreeTimeGS: Gaussian Primitives at Anytime Anywhere for 4D Scene Reconstruction

TEMPORAL GRADIENT VERSION - Proper GIFStream-style per-frame gradient accumulation.

Key improvements over canonical version:
1. Per-frame gradient accumulation using GLOBAL indices (all Gaussians, not just visible)
2. Visibility tracking via radii > 0 mask with proper index mapping
3. Custom densification using temporal combined gradients (Eq. 12 from GIFStream)
4. Proper handling of index mapping between visible and all Gaussians

CANONICAL PHASE IMPLEMENTATION with proper densification tracking for 4D Gaussians.

This implementation addresses the key challenges:
1. During densification (split/duplicate), temporal params (times, durations, velocities)
   are automatically preserved by gsplat's _update_param_with_optimizer
2. During canonical phase, ALL frames are sampled but motion is disabled
3. Temporal opacity is set to 1.0 during canonical to prevent incorrect pruning
4. Learning rates are carefully managed for position, velocity, and temporal params

================================================================================
THREE-PHASE TRAINING STRATEGY (GIFStream-style canonical phase)
================================================================================

Phase 1: CANONICAL (steps 0 to canonical_phase_steps)
    - ALL frames sampled (time t from data, not fixed!)
    - Temporal opacity = 1.0 (all Gaussians visible at all times)
    - Motion DISABLED: positions = means (velocity zeroed, no offset)
    - Densification ENABLED: split/clone/prune
    - Temporal params FROZEN: times, durations, velocities not updated
    - Goal: Learn a "consensus" static representation that fits ALL frames
    - Key insight: Model sees all timesteps but same static structure

Phase 2: TRANSITION (canonical_phase_steps to canonical_phase_steps + transition_phase_steps)
    - ALL frames sampled (same as canonical)
    - Motion ENABLED: positions = means + vel * (t - mu_t)
    - Temporal opacity = 1.0 still (adapt to motion first)
    - Temporal params BEGIN learning with low LR
    - Goal: Smooth transition from static to 4D

Phase 3: FULL 4D (after transition)
    - Time t from actual training data
    - Full motion model active
    - Temporal opacity computed from Gaussian durations
    - All parameters learning
    - Velocity LR annealing
    - Goal: Learn temporal dynamics

================================================================================
USAGE
================================================================================

# Quick test (100 steps)
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians_temporal.py fast \\
    --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
    --result-dir=/data/shared/elaheh/elly_test \\
    --start-frame=0 --end-frame=10 \\
    --max-steps=100

# Full training with temporal gradient densification
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians_temporal.py default \\
    --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
    --result-dir=/data/shared/elaheh/elly_temporal \\
    --start-frame=0 --end-frame=100 \\
    --max-steps=30000

# MCMC-style training (relocation, no split/clone)
CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_gaussians_temporal.py mcmc \\
    --data-dir=/data/shared/elaheh/4D_demo/elly/undistorted \\
    --result-dir=/data/shared/elaheh/elly_mcmc \\
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
from typing import Dict, List, Optional, Tuple, Union, Any

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
from gsplat.strategy import DefaultStrategy, MCMCStrategy
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
class FreeTimeCanonicalConfig:
    """Configuration for FreeTimeGS training with canonical phase densification."""

    # Data
    data_dir: str = "data/4d_scene"
    result_dir: str = "/data/shared/elaheh/elly_free_roma"
    data_factor: int = 1

    # Frame range
    start_frame: int = 0
    end_frame: int = 300
    frame_step: int = 1

    # Training
    max_steps: int = 30_000
    batch_size: int = 1
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 15_000, 30_000])

    # Evaluation speed options
    max_eval_samples: int = 5  # Max images to evaluate (0 = all). Speeds up eval with many val images
    skip_eval_render: bool = False  # Skip saving rendered images during eval (saves IO time)

    # Model
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    max_init_points: int = 600_000  # Allow more points for dense initialization
    init_opacity: float = 0.5
    init_scale: float = 1.0
    init_duration: float = 5.0  # Large initial duration - all visible initially

    # Loss weights
    lambda_img: float = 0.8
    lambda_ssim: float = 0.2
    lambda_perc: float = 0.01
    lambda_reg: float = 0.0  # 4D regularization (disabled for stability)

    # Velocity regularization - prevents velocities from growing unbounded
    # NOTE: velocity relates to displacement via: displacement = velocity * (t - μt)
    # Since (t - μt) ∈ [-0.5, 0.5], max_displacement ≈ velocity * 0.5
    # For a scene spanning ~10 units, velocity=2 means max displacement of 1 unit (10% of scene)
    # For mostly static scenes, use lower values (0.2-0.5)
    lambda_velocity: float = 0.01  # Penalize large velocity magnitudes
    velocity_max_magnitude: float = 0.5  # Soft cap (max displacement ≈ 0.25 units)
    velocity_grad_clip: float = 0.5  # Clip velocity gradients (conservative)

    # Duration regularization - prevents Gaussians from becoming too temporally narrow
    # Short durations cause Gaussians to appear/disappear quickly, hurting reconstruction
    # duration_min is in log-space: exp(duration_min) = actual min duration
    lambda_duration: float = 0.05  # Penalize durations below min threshold (was 0.001, too weak)
    duration_min: float = 0.5  # Min duration (actual, not log). Below this = penalty

    # Opacity regularization for DYNAMIC points only
    # Encourages dynamic Gaussians to maintain visible opacity
    lambda_dynamic_opacity: float = 0.01  # Encourage higher opacity for dynamic (was 0.0)

    # Temporal smoothness loss - penalizes large position changes across adjacent times
    # Prevents chaotic velocity patterns and encourages smooth motion
    lambda_temporal_smooth: float = 0.005  # Weight for temporal smoothness loss
    temporal_smooth_dt: float = 0.05  # Time delta for smoothness comparison

    # =========================================================================
    # FREETIMEGS-STYLE 4D TRAINING (from arxiv 2506.05348)
    # =========================================================================

    # 4D Regularization: ℒ_reg = (1/N) * Σ(σ * sg[σ(t)])
    # WARNING: This penalizes ALL visible Gaussians, hurting dynamic regions!
    # Disabled by default - analysis shows it causes dynamic Gaussians to be pruned
    lambda_4d_reg: float = 0.0  # DISABLED (was 0.01) - conflicts with dynamic opacity needs

    # =========================================================================
    # MOTION FACTOR (GIFStream-style velocity gating)
    # =========================================================================
    # Each Gaussian has a learned motion_factor that gates velocity contribution:
    # effective_velocity = sigmoid(motion_factor) * velocity * dt
    # This allows the model to learn which Gaussians should move vs stay static
    # - motion_factor = -inf → sigmoid = 0 → static Gaussian
    # - motion_factor = +inf → sigmoid = 1 → full velocity
    # GIFStream-style motion factor gating
    use_motion_factor: bool = True  # Enable motion factor gating
    motion_factor_lr: float = 1e-3  # Learning rate for motion factor
    motion_factor_init: float = -2.0  # Initial value (sigmoid(-2) ≈ 0.12, mostly static)

    # Factor regularization - encourages binary (0/1) motion factors
    # Loss = λ * factor * (1 - factor) which is 0 at 0 and 1, max at 0.5
    lambda_factor_sparsity: float = 0.01  # Encourage factors to be 0 or 1

    # =========================================================================
    # ROTATION VELOCITY (Angular velocity for rotating objects)
    # =========================================================================
    # q(t) = q_base * quaternion_exp(quat_velocity * dt)
    # This allows modeling rotating objects (spinning wheels, turning heads, etc.)
    use_rotation_velocity: bool = True  # Enable rotation velocity
    quat_velocity_lr: float = 5e-4  # Learning rate for quaternion velocity
    lambda_quat_velocity_reg: float = 0.001  # Regularize to prevent wild rotations

    # =========================================================================
    # PER-FRAME GRADIENT ACCUMULATION (GIFStream-style temporal-aware densification)
    # =========================================================================
    # GIFStream paper Eq. 12: combined_grad = α * max_t(g_t) + (1-α) * mean_t(g_t)
    # where g_t = grad_accum_t / visibility_count_t (normalized per frame)
    # This captures both peak gradients and average gradients across time
    #
    # This version uses GLOBAL indices for gradient tracking:
    # - radii > 0 gives visibility mask for ALL Gaussians (not just a subset)
    # - Gradients accumulated at global indices, not visible-only indices
    # - Custom densification extracts gradients for visible Gaussians
    use_per_frame_gradients: bool = True  # ENABLED - proper global index tracking
    per_frame_grad_peak_ratio: float = 0.3  # α in GIFStream formula (weight for max) - higher = more focus on peak times
    num_time_bins: int = 50  # Number of time bins (like GOP_size in GIFStream)
    densify_check_interval: int = 50  # Reset gradient accumulation every N steps (was 100)
    densify_visibility_threshold: float = 0.5  # Densify if visible >50% of check_interval (was 0.8)
    temporal_grad_threshold: float = 0.0001  # Lower threshold = more aggressive densification (was 0.0002)

    # Periodic Relocation: Move low-opacity Gaussians to high-importance regions
    # This helps densify moving objects that were missed during initialization
    # Similar to MCMC relocation but with 4D-aware sampling scores
    periodic_relocation_enabled: bool = True
    periodic_relocation_every: int = 100  # Relocate every N steps
    periodic_relocation_opacity_threshold: float = 0.01  # Relocate if opacity below this

    # Skip canonical phase - train with varying times from the START
    # FreeTimeGS paper does NOT use canonical phase - each Gaussian has its own time
    # This avoids the problem of missing moving objects at fixed t=0.5
    skip_canonical_phase: bool = False  # Set True for FreeTimeGS-style training

    # =========================================================================
    # CANONICAL PHASE CONFIGURATION (used if skip_canonical_phase=False)
    # =========================================================================

    # Canonical phase: treat scene as static, run densification
    # During canonical phase (GIFStream-style):
    # - ALL frames sampled (time t from data, not fixed!)
    # - Temporal opacity = 1.0 (all Gaussians visible for fair pruning)
    # - Motion DISABLED (velocity zeroed, positions = means)
    # - Densification runs normally (split/clone/prune)
    # - Temporal params frozen
    # - Goal: Learn "consensus" static representation across all times
    canonical_phase_steps: int = 5000
    canonical_time: float = 0.5  # DEPRECATED: no longer used, kept for compatibility

    # Transition phase: motion still disabled but preparing for 4D
    transition_phase_steps: int = 2000

    # 4D warmup: first N steps of 4D phase use motion but keep temporal_opacity = 1.0
    # This allows model to adapt to motion before also dealing with temporal visibility
    # After this warmup, full temporal_opacity is applied
    four_d_warmup_steps: int = 2000

    # Densification strategy during canonical phase
    # "default" uses DefaultStrategy (split/clone/prune)
    # "mcmc" uses MCMCStrategy (relocation only)
    densification_mode: str = "default"

    # Densification parameters
    densify_every: int = 200  # Densify every 200 steps (was 500) - more frequent for temporal
    densify_start_iter: int = 1000  # Start after warmup
    densify_stop_iter: int = 25_000
    prune_opacity_threshold: float = 0.005
    grow_grad2d_threshold: float = 0.0002
    grow_scale3d_threshold: float = 0.01
    reset_opacity_every: int = 3000  # Set to very high to disable
    # Maximum number of Gaussians (applies to both DefaultStrategy and MCMC)
    max_gaussians: int = 500_000  # Cap at 500k to prevent memory issues
    # Stop densification after entering 4D phase (only densify during canonical)
    # This prevents explosion of Gaussians during 4D training
    densify_only_canonical: bool = True  # Recommended: only densify during canonical phase

    # Temporal-aware pruning (for 4D phase)
    # Sample opacity at multiple time points and only prune if low at ALL times
    # This prevents pruning Gaussians that are important at specific times
    temporal_aware_pruning: bool = True  # Enable temporal-aware pruning
    temporal_prune_times: int = 5  # Number of time points to sample [0, 0.25, 0.5, 0.75, 1.0]

    # MCMC-specific parameters (mcmc_cap_max is deprecated, use max_gaussians instead)
    mcmc_cap_max: int = 500_000  # Will be overridden by max_gaussians
    mcmc_noise_lr: float = 5e5
    mcmc_min_opacity: float = 0.005

    # =========================================================================
    # LEARNING RATE CONFIGURATION
    # =========================================================================

    # Position LR: Same as simple_trainer.py (1.6e-4)
    # During canonical phase, positions train normally like standard 3DGS
    # After canonical, velocity handles motion while positions are anchor points
    position_lr: float = 1.6e-4  # Multiplied by scene_scale (same as simple_trainer.py)

    # Velocity LR: Uses annealing schedule
    # LOWERED from 5e-3 to 1e-3 to prevent velocity explosion
    velocity_lr_start: float = 1e-3
    velocity_lr_end: float = 1e-5

    # GIFStream-style: Train velocity during canonical but zero its contribution
    # This allows velocity to learn motion patterns while keeping positions static
    # When 4D phase starts, velocity is already "pre-trained" like GIFStream's motion MLP
    train_velocity_during_canonical: bool = True  # NEW: Train vel but zero output
    velocity_canonical_lr_scale: float = 0.5  # LR scale during canonical (lower than full 4D)

    # Initialize velocity from gradient statistics after canonical phase
    # High gradient regions likely have motion -> initialize with higher velocity capacity
    init_velocity_from_gradients: bool = True  # NEW: GIFStream-style initialization
    velocity_init_step: int = -1  # Step to initialize (-1 = auto after canonical)

    # Temporal params LR
    times_lr: float = 5e-4
    durations_lr: float = 5e-4

    # Appearance params LR (matching simple_trainer.py)
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2  # Same as simple_trainer.py (was 5e-3, too low!)
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    # Freeze temporal params completely (just use initialization)
    freeze_temporal_params: bool = False  # Now False - we train temporal params after canonical

    # =========================================================================
    # RENDERING CONFIGURATION
    # =========================================================================
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    antialiased: bool = False
    # Minimum scale for Gaussians (in world units, after exp)
    # Prevents extremely small Gaussians from becoming invisible black dots
    # Typical value: 1e-4 to 1e-3 depending on scene scale
    scale_min: float = 1e-4

    # =========================================================================
    # MISC
    # =========================================================================
    test_every: int = 8
    tb_every: int = 100
    tb_image_every: int = 200
    disable_viewer: bool = True
    port: int = 8080

    # Initialization mode
    # "multiframe" - loads ALL points from ALL frames (~500k+, dense)
    # "grid_tracked" - grid-based stratified sampling with Roma triangulation (~10k-20k, sparse but good velocity)
    init_mode: str = "grid_tracked"  # Use Roma-triangulated points like working versions
    use_velocity_init: bool = True
    velocity_npz_path: Optional[str] = None
    knn_match_threshold: float = 0.5

    # =========================================================================
    # ALTERNATING FREEZE CONFIGURATION
    # =========================================================================
    # Enable alternating freeze between velocity and position/opacity
    # This stabilizes training by preventing both from changing simultaneously
    # NOTE: Only active AFTER canonical phase (during 4D training)
    alternating_freeze_enabled: bool = True
    # Number of steps to train velocity while freezing position/opacity
    alternating_velocity_steps: int = 2000
    # Number of steps to train position/opacity while freezing velocity
    alternating_appearance_steps: int = 2000
    # Step to start alternating freeze (after canonical phase ends)
    # Set to -1 to auto-start after canonical_phase_steps
    alternating_freeze_start: int = -1  # Will be set to canonical_phase_steps
    # Step to stop alternating freeze (run until end of training)
    alternating_freeze_stop: int = 30000
    # Only densify during position phase (when positions are trainable)
    densify_only_in_position_phase: bool = True
    # Start with POSITION phase (recommended for 4D warmup - adapt positions to varying t first)
    alternating_start_with_position: bool = True

    # =========================================================================
    # WARMUP PHASE CONFIGURATION
    # =========================================================================
    # Number of warmup steps with no densification/pruning at start
    warmup_steps: int = 1000

    # =========================================================================
    # VELOCITY-BASED STATIC CLASSIFICATION
    # =========================================================================
    # At this step, classify Gaussians by velocity magnitude
    # Low-velocity points become permanently static
    # Set to -1 to auto-trigger after first position phase in 4D
    velocity_classification_step: int = -1  # Auto: after first POS phase in 4D
    # Percentage of points to keep as STATIC (lowest velocity points)
    velocity_static_percentile: float = 70.0  # 70% static, 30% dynamic
    # Freeze temporal params for static points (velocity=0, duration=inf)
    # Static points can still update opacity/scale but NOT temporal info
    freeze_static_temporal_params: bool = True

    # =========================================================================
    # DEBUG CONFIGURATION
    # =========================================================================
    debug_enabled: bool = True
    debug_every: int = 100  # Check every N steps
    debug_print_stats: bool = True  # Print parameter stats
    debug_check_nan: bool = True  # Check for NaN/Inf
    debug_check_gradients: bool = True  # Check gradient magnitudes
    debug_check_lr: bool = True  # Print learning rates
    debug_alert_threshold: float = 100.0  # Alert if any value exceeds this


def create_freetime_splats_with_optimizers(
    parser: FreeTimeParser,
    cfg: FreeTimeCanonicalConfig,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    """
    Create FreeTimeGS splats with 4D parameters and optimizers.

    Each Gaussian has:
    - means: [N, 3] position µx (anchor position)
    - times: [N, 1] temporal center µt
    - durations: [N, 1] temporal spread s (log scale)
    - velocities: [N, 3] velocity v
    - scales: [N, 3] spatial scale (log scale)
    - quats: [N, 4] orientation quaternion
    - opacities: [N] opacity σ (logit scale)
    - sh0: [N, 1, 3] DC spherical harmonics
    - shN: [N, K, 3] higher-order SH coefficients
    """
    parser_transform = parser.transform if hasattr(parser, 'transform') else None

    # Load points based on initialization mode
    if cfg.init_mode == "grid_tracked":
        print(f"[FreeTimeGS] Using grid-tracked initialization (recommended)")
        init_data = load_multiframe_colmap_grid_tracked(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            frame_step=cfg.frame_step,
            grid_divisions=(10, 10, 4),
            max_points_per_cell=2000,
            match_threshold=0.1,
            max_error=1.0,
            transform=parser_transform,
        )
    elif cfg.init_mode == "multiframe":
        print(f"[FreeTimeGS] Using multiframe COLMAP initialization")
        init_data = load_multiframe_colmap_points(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            frame_step=cfg.frame_step,
            max_error=1.0,
            match_threshold=0.1,
            transform=parser_transform,
        )
    elif cfg.init_mode == "single_frame":
        print(f"[FreeTimeGS] Using single-frame initialization")
        init_data = load_single_frame_with_velocity(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            reference_time=0.5,
            max_error=2.0,
            match_threshold=0.1,
            transform=parser_transform,
        )
    elif cfg.init_mode == "startframe":
        print(f"[FreeTimeGS] Using startframe + tracked velocity initialization")
        init_data = load_startframe_tracked_velocity(
            cfg.data_dir,
            start_frame=cfg.start_frame,
            end_frame=cfg.end_frame,
            frame_step=cfg.frame_step,
            max_error=1.0,
            match_threshold=None,
            transform=parser_transform,
        )
    else:
        # Reference mode
        print(f"[FreeTimeGS] Using reference COLMAP initialization")
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
        N = points.shape[0]
        init_data = {
            'positions': points,
            'times': torch.rand((N, 1)),
            'velocities': torch.zeros((N, 3)),
            'colors': rgbs,
            'has_velocity': torch.zeros(N, dtype=torch.bool),
        }

    points = init_data['positions']
    times = init_data['times']
    velocities = init_data['velocities']
    rgbs = init_data['colors']
    has_velocity = init_data['has_velocity']

    N = points.shape[0]
    print(f"[FreeTimeGS] Loaded {N} points")
    print(f"[FreeTimeGS] Points with velocity: {has_velocity.sum().item()} ({100*has_velocity.float().mean():.1f}%)")

    # Filter outliers
    scene_scale = parser.scene_scale
    max_point_dist = 5.0 * scene_scale
    point_dists = torch.norm(points, dim=1)
    valid_points = point_dists < max_point_dist

    points = points[valid_points]
    rgbs = rgbs[valid_points]
    times = times[valid_points]
    velocities = velocities[valid_points]
    has_velocity = has_velocity[valid_points]

    print(f"[FreeTimeGS] After distance filtering: {len(points)} points")

    # Subsample if needed
    if cfg.max_init_points > 0 and len(points) > cfg.max_init_points:
        idx = torch.randperm(len(points))[:cfg.max_init_points]
        points = points[idx]
        rgbs = rgbs[idx]
        times = times[idx]
        velocities = velocities[idx]
        has_velocity = has_velocity[idx]
        print(f"[FreeTimeGS] Subsampled to {cfg.max_init_points} points")

    N = points.shape[0]

    # Initialize scales from KNN - IMPORTANT: compute PER TIME BUCKET
    # This prevents moving objects from getting oversized scales
    # (Without this, KNN neighbors span temporal frames, giving biased large scales)
    unique_times = torch.unique(times)
    scales = torch.zeros(N, 3)

    if len(unique_times) > 1:
        # Multiple time steps - compute KNN per time bucket
        print(f"[FreeTimeGS] Computing scales per time bucket ({len(unique_times)} buckets)...")
        for t in unique_times:
            mask = (times == t).squeeze()
            if mask.sum() < 4:
                # Not enough points, use global fallback
                continue
            pts_t = points[mask]
            dist2_avg_t = (knn(pts_t, min(4, len(pts_t)))[:, 1:] ** 2).mean(dim=-1)
            dist_avg_t = torch.sqrt(torch.clamp(dist2_avg_t, min=1e-6))
            scales[mask] = torch.log(dist_avg_t * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

        # Handle any points that weren't processed (tiny buckets)
        unprocessed = (scales.abs().sum(dim=-1) == 0)
        if unprocessed.sum() > 0:
            dist2_avg = (knn(points[unprocessed], 4)[:, 1:] ** 2).mean(dim=-1)
            dist_avg = torch.sqrt(torch.clamp(dist2_avg, min=1e-6))
            scales[unprocessed] = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)
    else:
        # Single time step - use global KNN
        dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(torch.clamp(dist2_avg, min=1e-6))
        scales = torch.log(dist_avg * cfg.init_scale).unsqueeze(-1).repeat(1, 3)

    # Distribute across ranks
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]
    times = times[world_rank::world_size]
    velocities = velocities[world_rank::world_size]
    N = points.shape[0]

    # Clamp initial velocities to reasonable range based on scene
    # Max displacement should be a fraction of scene size
    # displacement = velocity * delta_t, where delta_t ∈ [-0.5, 0.5]
    pos_range = points.max() - points.min()
    max_reasonable_velocity = cfg.velocity_max_magnitude * 2  # Allow 2x soft cap at init
    vel_mag = velocities.norm(dim=-1, keepdim=True)
    vel_scale = torch.clamp(max_reasonable_velocity / (vel_mag + 1e-8), max=1.0)
    velocities = velocities * vel_scale
    clamped_count = (vel_mag.squeeze() > max_reasonable_velocity).sum().item()
    if clamped_count > 0:
        print(f"[FreeTimeGS] Clamped {clamped_count} velocities to max {max_reasonable_velocity:.2f}")

    # Initialize other parameters
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), cfg.init_opacity))
    durations = torch.log(torch.full((N, 1), cfg.init_duration))

    # Spherical harmonics
    colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    colors[:, 0, :] = rgb_to_sh(rgbs)

    # GIFStream-style "factors" tensor combining motion_factor and quat_velocity
    # factors[:, 0] = motion_factor (gates velocity contribution)
    # factors[:, 1:5] = quat_velocity (rotation angular velocity)
    # Using a combined tensor is more compatible with gsplat's densification
    factors = torch.zeros((N, 5))
    factors[:, 0] = cfg.motion_factor_init  # motion_factor initialization

    # Build parameters - core parameters always included
    params = [
        ("means", torch.nn.Parameter(points), cfg.position_lr * scene_scale),
        ("times", torch.nn.Parameter(times), cfg.times_lr),
        ("durations", torch.nn.Parameter(durations), cfg.durations_lr),
        ("velocities", torch.nn.Parameter(velocities), cfg.velocity_lr_start),
        ("factors", torch.nn.Parameter(factors), cfg.motion_factor_lr),  # Combined factors tensor
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
        ("sh0", torch.nn.Parameter(colors[:, :1, :]), cfg.sh0_lr),
        ("shN", torch.nn.Parameter(colors[:, 1:, :]), cfg.shN_lr),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    # Print initialization statistics
    print("\n" + "="*70)
    print("[INIT] Gaussian initialization statistics:")
    print("="*70)
    print(f"  Number of Gaussians: {N}")
    print(f"  Positions: min={points.min():.4f}, max={points.max():.4f}")
    print(f"  Times: min={times.min():.4f}, max={times.max():.4f}, mean={times.mean():.4f}")
    print(f"  Durations (exp): min={torch.exp(durations).min():.4f}, max={torch.exp(durations).max():.4f}")
    print(f"  Velocities: mean_norm={velocities.norm(dim=1).mean():.4f}")
    print(f"  Motion factors: init={cfg.motion_factor_init:.2f}, sigmoid={torch.sigmoid(factors[:, 0]).mean():.4f}, enabled={cfg.use_motion_factor}")
    print(f"  Quat velocities: mean_norm={factors[:, 1:5].norm(dim=1).mean():.4f}, enabled={cfg.use_rotation_velocity}")
    print(f"  Scene scale: {scene_scale:.4f}")
    print(f"  Features enabled: motion_factor={cfg.use_motion_factor}, rotation_vel={cfg.use_rotation_velocity}")
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


class FreeTime4DStrategy:
    """
    Custom strategy wrapper that handles 4D Gaussians.

    Key features:
    1. During canonical phase, uses temporal_opacity=1.0 for fair pruning
    2. Properly tracks temporal params during densification
    3. Integrates with either DefaultStrategy or MCMCStrategy
    """

    def __init__(
        self,
        cfg: FreeTimeCanonicalConfig,
        base_strategy: Union[DefaultStrategy, MCMCStrategy],
        device: str = "cuda",
    ):
        self.cfg = cfg
        self.base_strategy = base_strategy
        self.device = device

        # MCMC binomial coefficients for relocation
        if isinstance(base_strategy, MCMCStrategy):
            n_max = 51
            self.binoms = torch.zeros((n_max, n_max))
            for n in range(n_max):
                for k in range(n + 1):
                    self.binoms[n, k] = math.comb(n, k)
            self.binoms = self.binoms.to(device)

    def initialize_state(self, scene_scale: float, num_gaussians: int = None) -> Dict[str, Any]:
        """Initialize strategy state including per-frame gradient tracking."""
        if isinstance(self.base_strategy, DefaultStrategy):
            state = self.base_strategy.initialize_state(scene_scale=scene_scale)
        else:
            state = self.base_strategy.initialize_state()

        # GIFStream-style per-frame gradient accumulation
        # Uses fixed-size buffers indexed by time bin for proper temporal tracking
        if self.cfg.use_per_frame_gradients:
            # These will be lazily initialized when we know the number of Gaussians
            state["grad2d_accum"] = None  # [N, num_time_bins] - accumulated 2D gradients
            state["grad2d_count"] = None  # [N, num_time_bins] - visibility count per time
            state["opacity_accum"] = None  # [N, num_time_bins] - accumulated opacity for pruning
            state["num_time_bins"] = self.cfg.num_time_bins
            state["last_reset_step"] = 0  # Track when we last reset accumulation

        return state

    def check_sanity(
        self,
        splats: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Check that all required parameters are present."""
        # Check base requirements
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in splats, f"{key} is required but missing"
        # Check 4D requirements
        for key in ["times", "durations", "velocities"]:
            assert key in splats, f"{key} is required for 4D but missing"
        # Check GIFStream-style combined factors tensor (motion_factor + quat_velocity)
        # factors[:, 0] = motion_factor, factors[:, 1:5] = quat_velocity
        cfg = self.cfg
        if cfg.use_motion_factor or cfg.use_rotation_velocity:
            assert "factors" in splats, "factors tensor required when use_motion_factor or use_rotation_velocity=True"

    def step_pre_backward(
        self,
        params: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """Pre-backward hook - mainly for gradient tracking."""
        if isinstance(self.base_strategy, DefaultStrategy):
            self.base_strategy.step_pre_backward(
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
                info=info,
            )

    def update_per_frame_gradients(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
        current_time: float,
        num_gaussians: int,
        step: int,
    ):
        """
        Update per-frame gradient accumulation (GIFStream-style Eq. 12).

        GIFStream paper formula:
        g_t = grad_accum_t / visibility_count_t  (normalize per frame)
        combined_grad = α * max_t(g_t) + (1-α) * mean_t(g_t)

        This version uses GLOBAL indices:
        - info["radii"] has shape [N_total] with values > 0 for visible Gaussians
        - info["means2d"] has shape [N_visible, 2] (only visible Gaussians)
        - We use radii > 0 to get global indices and accumulate properly

        This captures both:
        - Peak gradients (important for densifying moving objects at specific times)
        - Average gradients (overall importance across all times)
        """
        cfg = self.cfg
        if not cfg.use_per_frame_gradients:
            return

        # Extract 2D gradient from info (from rasterization)
        if "means2d" not in info or "radii" not in info:
            return

        means2d = info["means2d"]  # [N_visible, 2]
        radii = info["radii"]  # [N_total] - visibility mask

        if means2d.grad is None:
            return

        device = means2d.device
        num_time_bins = state["num_time_bins"]

        # Get visibility mask and global indices
        # radii > 0 indicates visible Gaussians
        visible_mask = radii.squeeze() > 0  # [N_total] boolean
        visible_indices = torch.where(visible_mask)[0]  # Global indices of visible Gaussians
        n_visible = visible_indices.shape[0]

        # Sanity check
        if n_visible != means2d.shape[0]:
            # Mismatch - skip this step (can happen with packed mode)
            if step % 1000 == 0:
                print(f"[GIFStream] Warning: visibility mismatch at step {step}: "
                      f"radii>0 gives {n_visible}, means2d has {means2d.shape[0]}")
            return

        # Lazy initialization of state buffers
        if state["grad2d_accum"] is None:
            state["grad2d_accum"] = torch.zeros((num_gaussians, num_time_bins), device=device)
            state["grad2d_count"] = torch.zeros((num_gaussians, num_time_bins), device=device)
            state["opacity_accum"] = torch.zeros((num_gaussians, num_time_bins), device=device)
            print(f"[GIFStream] Initialized per-frame gradient buffers: [{num_gaussians}, {num_time_bins}]")

        # Handle buffer size changes (after densification)
        if state["grad2d_accum"].shape[0] != num_gaussians:
            old_n = state["grad2d_accum"].shape[0]
            if num_gaussians > old_n:
                # Expand buffers
                extra = num_gaussians - old_n
                state["grad2d_accum"] = torch.cat([
                    state["grad2d_accum"],
                    torch.zeros((extra, num_time_bins), device=device)
                ], dim=0)
                state["grad2d_count"] = torch.cat([
                    state["grad2d_count"],
                    torch.zeros((extra, num_time_bins), device=device)
                ], dim=0)
                state["opacity_accum"] = torch.cat([
                    state["opacity_accum"],
                    torch.zeros((extra, num_time_bins), device=device)
                ], dim=0)
            else:
                # Shrink buffers (after pruning)
                state["grad2d_accum"] = state["grad2d_accum"][:num_gaussians]
                state["grad2d_count"] = state["grad2d_count"][:num_gaussians]
                state["opacity_accum"] = state["opacity_accum"][:num_gaussians]

        # Convert time (0-1) to time bin index (0 to num_time_bins-1)
        time_idx = int(current_time * (num_time_bins - 1))
        time_idx = max(0, min(time_idx, num_time_bins - 1))

        # Compute gradient magnitude per VISIBLE Gaussian
        grad_2d = means2d.grad.detach().norm(dim=-1)  # [N_visible]

        # Accumulate gradients at GLOBAL indices using index_add_
        # This properly maps visible gradients to the global buffer
        state["grad2d_accum"][:, time_idx].index_add_(0, visible_indices, grad_2d)
        state["grad2d_count"][:, time_idx].index_add_(
            0, visible_indices, torch.ones(n_visible, device=device)
        )

        # Accumulate opacity for pruning decisions (if available)
        # Note: opacities might be for visible only or all - check shape
        if "opacities" in info:
            opacities = info["opacities"]
            if opacities.shape[0] == n_visible:
                # Opacity for visible Gaussians only
                opacity = torch.sigmoid(opacities).detach().squeeze(-1)  # [N_visible]
                state["opacity_accum"][:, time_idx].index_add_(0, visible_indices, opacity)
            elif opacities.shape[0] == num_gaussians:
                # Opacity for all Gaussians - extract visible ones
                opacity = torch.sigmoid(opacities[visible_mask]).detach().squeeze(-1)  # [N_visible]
                state["opacity_accum"][:, time_idx].index_add_(0, visible_indices, opacity)

    def compute_combined_gradient(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Compute combined gradient using GIFStream formula (Eq. 12).

        Formula: combined = α * max_t(g_t) + (1-α) * mean_t(g_t)
        where g_t = grad_accum_t / visibility_count_t

        Returns:
            combined_grad: [N] tensor of combined gradient magnitudes
        """
        cfg = self.cfg
        if not cfg.use_per_frame_gradients or state["grad2d_accum"] is None:
            return None

        # Normalize gradients by visibility count per frame
        # g_t = grad_accum_t / count_t
        per_frame_grads = state["grad2d_accum"] / (state["grad2d_count"] + 1e-7)  # [N, T]
        per_frame_grads[per_frame_grads.isnan()] = 0.0

        # GIFStream formula: α * max_t(g_t) + (1-α) * mean_t(g_t)
        alpha = cfg.per_frame_grad_peak_ratio
        max_grad = per_frame_grads.max(dim=-1)[0]  # [N]
        mean_grad = per_frame_grads.mean(dim=-1)  # [N]

        combined = alpha * max_grad + (1 - alpha) * mean_grad
        return combined

    def compute_visibility_score(self, state: Dict[str, Any]) -> torch.Tensor:
        """
        Compute visibility score for pruning decisions.

        Uses same formula as gradients: α * max_t(count_t) + (1-α) * mean_t(count_t)
        """
        cfg = self.cfg
        if not cfg.use_per_frame_gradients or state["grad2d_count"] is None:
            return None

        alpha = cfg.per_frame_grad_peak_ratio
        max_count = state["grad2d_count"].max(dim=-1)[0]  # [N]
        mean_count = state["grad2d_count"].mean(dim=-1)  # [N]

        visibility = alpha * max_count + (1 - alpha) * mean_count
        return visibility

    def reset_gradient_accumulation(self, state: Dict[str, Any], step: int):
        """Reset gradient accumulation buffers (called after densification)."""
        if not self.cfg.use_per_frame_gradients:
            return

        if state["grad2d_accum"] is not None:
            state["grad2d_accum"].zero_()
            state["grad2d_count"].zero_()
            state["opacity_accum"].zero_()
            state["last_reset_step"] = step

    def get_combined_grad_for_densification(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Replace the single-frame gradient with combined per-frame gradient for densification.

        Returns modified info dict with combined gradient if available.
        """
        cfg = self.cfg
        if not cfg.use_per_frame_gradients:
            return info

        # Compute combined gradient using GIFStream formula
        combined_grad = self.compute_combined_gradient(state)
        if combined_grad is None:
            return info

        # Create a copy of info with combined gradient
        info_modified = dict(info)
        info_modified["combined_grad_2d"] = combined_grad

        # Also add visibility score for pruning
        visibility_score = self.compute_visibility_score(state)
        if visibility_score is not None:
            info_modified["visibility_score"] = visibility_score

        return info_modified

    def step_post_backward(
        self,
        params: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        in_canonical_phase: bool = False,
        lr: float = None,  # For MCMCStrategy
    ):
        """
        Post-backward hook - densification.

        When use_per_frame_gradients=True:
        - Uses custom temporal densification based on accumulated per-frame gradients
        - GIFStream formula: combined = α * max_t(g_t) + (1-α) * mean_t(g_t)
        - Densifies based on temporal gradient threshold

        When use_per_frame_gradients=False:
        - Falls back to DefaultStrategy with single-frame gradients
        """
        cfg = self.cfg

        # Skip if outside densification range
        if step < cfg.densify_start_iter or step >= cfg.densify_stop_iter:
            return

        if cfg.use_per_frame_gradients:
            # Custom temporal densification using accumulated per-frame gradients
            if step % cfg.densify_every == 0:
                self._temporal_densification(params, optimizers, state, step, info)

            # Reset gradient accumulation after check interval
            if step % cfg.densify_check_interval == 0:
                self.reset_gradient_accumulation(state, step)
        elif isinstance(self.base_strategy, DefaultStrategy):
            # DefaultStrategy handles split/duplicate/prune automatically
            # The ops in gsplat handle ALL params in the dict, including our 4D params
            self.base_strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=state,
                step=step,
                info=info,
                packed=cfg.packed,
            )

        if isinstance(self.base_strategy, MCMCStrategy):
            # MCMCStrategy: relocate + add + noise
            self.binoms = self.binoms.to(params["means"].device)

            if step % cfg.densify_every == 0:
                # Relocate dead Gaussians
                opacities = torch.sigmoid(params["opacities"].flatten())
                dead_mask = opacities <= cfg.mcmc_min_opacity
                n_dead = dead_mask.sum().item()

                if n_dead > 0:
                    self._relocate_4d(params, optimizers, state, dead_mask)
                    print(f"[MCMC] Step {step}: Relocated {n_dead} Gaussians")

                # Add new Gaussians (grow by 5% up to cap)
                current_n = len(params["means"])
                n_target = min(cfg.mcmc_cap_max, int(1.05 * current_n))
                n_add = max(0, n_target - current_n)

                if n_add > 0:
                    self._sample_add_4d(params, optimizers, state, n_add)
                    print(f"[MCMC] Step {step}: Added {n_add} Gaussians, now {len(params['means'])}")

            # Add position noise for MCMC exploration
            if lr is not None:
                from gsplat.strategy.ops import inject_noise_to_position
                inject_noise_to_position(
                    params=params,
                    optimizers=optimizers,
                    state={},
                    scaler=lr * cfg.mcmc_noise_lr,
                )

    @torch.no_grad()
    def _temporal_densification(
        self,
        params: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ):
        """
        Custom densification using GIFStream-style temporal gradients.

        GIFStream formula (Eq. 12):
        combined_grad = α * max_t(g_t) + (1-α) * mean_t(g_t)
        where g_t = grad_accum_t / visibility_count_t

        This method:
        1. Computes combined gradient from accumulated per-frame gradients
        2. Identifies high-gradient Gaussians for split (large scale) or clone (small scale)
        3. Identifies low-opacity Gaussians for pruning
        4. Uses gsplat's ops to perform densification
        """
        from gsplat.strategy.ops import split, duplicate, remove

        cfg = self.cfg
        device = params["means"].device
        n_gaussians = len(params["means"])

        # Check Gaussian cap
        if n_gaussians >= cfg.max_gaussians:
            print(f"[Temporal] Step {step}: At max Gaussians ({n_gaussians}), skipping grow")
            return

        # Compute combined gradient using GIFStream formula
        combined_grad = self.compute_combined_gradient(state)
        if combined_grad is None:
            print(f"[Temporal] Step {step}: No gradients accumulated yet, skipping")
            return

        # Ensure combined_grad matches current params
        if combined_grad.shape[0] != n_gaussians:
            print(f"[Temporal] Step {step}: Gradient buffer mismatch "
                  f"({combined_grad.shape[0]} vs {n_gaussians}), skipping")
            return

        # Get visibility score for pruning decisions
        visibility_score = self.compute_visibility_score(state)

        # ===== IDENTIFY GAUSSIANS FOR SPLIT/CLONE =====
        # High gradient means underfit - need more Gaussians
        high_grad_mask = combined_grad > cfg.temporal_grad_threshold

        # Scale-based split vs clone decision
        # Large Gaussians -> split (divide in half)
        # Small Gaussians -> clone (duplicate)
        scales = torch.exp(params["scales"])  # [N, 3]
        scale_max = scales.max(dim=-1)[0]  # [N]
        is_large = scale_max > cfg.grow_scale3d_threshold

        split_mask = high_grad_mask & is_large
        clone_mask = high_grad_mask & ~is_large

        # ===== IDENTIFY GAUSSIANS FOR PRUNING =====
        # Low opacity AND not visible much -> prune
        opacities = torch.sigmoid(params["opacities"].flatten())  # [N]
        low_opacity_mask = opacities < cfg.prune_opacity_threshold

        # Also check visibility - don't prune if visible at some times
        if visibility_score is not None:
            # If visibility score is very low, Gaussian was rarely seen
            min_visibility = cfg.densify_visibility_threshold * cfg.densify_check_interval
            rarely_visible = visibility_score < min_visibility
            prune_mask = low_opacity_mask & rarely_visible
        else:
            prune_mask = low_opacity_mask

        # Count operations
        n_split = split_mask.sum().item()
        n_clone = clone_mask.sum().item()
        n_prune = prune_mask.sum().item()

        # Cap growth to prevent explosion
        max_new = cfg.max_gaussians - n_gaussians
        if n_split + n_clone > max_new:
            # Prioritize splits over clones
            if n_split > max_new:
                # Too many splits - randomly select
                split_indices = split_mask.nonzero(as_tuple=True)[0]
                keep = torch.randperm(n_split, device=device)[:max_new]
                split_mask = torch.zeros_like(split_mask)
                split_mask[split_indices[keep]] = True
                clone_mask = torch.zeros_like(clone_mask)
                n_split = max_new
                n_clone = 0
            else:
                # Limit clones
                remaining = max_new - n_split
                clone_indices = clone_mask.nonzero(as_tuple=True)[0]
                keep = torch.randperm(n_clone, device=device)[:remaining]
                new_clone_mask = torch.zeros_like(clone_mask)
                new_clone_mask[clone_indices[keep]] = True
                clone_mask = new_clone_mask
                n_clone = remaining

        # Execute operations in order: split -> clone -> prune
        # This order is important because indices change after each op

        if n_split > 0:
            print(f"[Temporal] Step {step}: Splitting {n_split} Gaussians")
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=split_mask,
                revised_opacity=True,  # Reduce opacity after split
            )

        if n_clone > 0:
            print(f"[Temporal] Step {step}: Cloning {n_clone} Gaussians")
            duplicate(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=clone_mask,
            )

        # Recompute prune mask after growth (indices may have changed)
        if n_prune > 0:
            # Recompute since we may have added Gaussians
            n_after_grow = len(params["means"])
            opacities = torch.sigmoid(params["opacities"].flatten())

            # Simple low-opacity pruning after growth
            # Original Gaussians are at the start, new ones at the end
            # Only prune from original indices to avoid pruning just-added Gaussians
            prune_mask_new = torch.zeros(n_after_grow, dtype=torch.bool, device=device)
            prune_mask_new[:n_gaussians] = opacities[:n_gaussians] < cfg.prune_opacity_threshold

            n_prune_actual = prune_mask_new.sum().item()
            if n_prune_actual > 0:
                print(f"[Temporal] Step {step}: Pruning {n_prune_actual} Gaussians")
                remove(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    mask=prune_mask_new,
                )

        # Summary
        n_final = len(params["means"])
        print(f"[Temporal] Step {step}: {n_gaussians} -> {n_final} Gaussians "
              f"(split={n_split}, clone={n_clone})")

    @torch.no_grad()
    def _relocate_4d(
        self,
        params: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        dead_mask: Tensor,
    ):
        """
        Relocate dead Gaussians to positions of alive ones.
        Uses the same logic as gsplat's relocate but handles all 4D params.
        """
        from gsplat.relocation import compute_relocation

        opacities = torch.sigmoid(params["opacities"])
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = (~dead_mask).nonzero(as_tuple=True)[0]
        n = len(dead_indices)

        if n == 0 or len(alive_indices) == 0:
            return

        # Sample source Gaussians from alive ones (weighted by opacity)
        probs = opacities[alive_indices].flatten()
        probs = probs / probs.sum().clamp_min(1e-8)
        sampled_idxs = torch.multinomial(probs, n, replacement=True)
        sampled_idxs = alive_indices[sampled_idxs]

        # Compute new opacities and scales
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs, minlength=len(opacities))[sampled_idxs] + 1,
            binoms=self.binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=0.999, min=self.cfg.mcmc_min_opacity)

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "opacities":
                p[sampled_idxs] = torch.logit(new_opacities)
            elif name == "scales":
                p[sampled_idxs] = torch.log(new_scales)
            elif name == "means":
                # Add small noise to positions
                noise = torch.randn(n, 3, device=p.device) * 0.01
                p[dead_indices] = p[sampled_idxs] + noise
                return torch.nn.Parameter(p, requires_grad=p.requires_grad)
            # For ALL other params (including times, durations, velocities), copy from source
            p[dead_indices] = p[sampled_idxs]
            return torch.nn.Parameter(p, requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v[sampled_idxs] = 0  # Reset optimizer state for modified Gaussians
            return v

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

    @torch.no_grad()
    def _sample_add_4d(
        self,
        params: torch.nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        n: int,
    ):
        """
        Add new Gaussians by sampling from existing ones.
        Handles all 4D parameters.
        """
        from gsplat.relocation import compute_relocation

        opacities = torch.sigmoid(params["opacities"])

        # Sample source Gaussians
        probs = opacities.flatten()
        probs = probs / probs.sum().clamp_min(1e-8)
        sampled_idxs = torch.multinomial(probs, n, replacement=True)

        # Compute new opacities and scales
        new_opacities, new_scales = compute_relocation(
            opacities=opacities[sampled_idxs],
            scales=torch.exp(params["scales"])[sampled_idxs],
            ratios=torch.bincount(sampled_idxs, minlength=len(opacities))[sampled_idxs] + 1,
            binoms=self.binoms,
        )
        new_opacities = torch.clamp(new_opacities, max=0.999, min=self.cfg.mcmc_min_opacity)

        def param_fn(name: str, p: Tensor) -> Tensor:
            if name == "opacities":
                p[sampled_idxs] = torch.logit(new_opacities)
                p_new = p[sampled_idxs]
            elif name == "scales":
                p[sampled_idxs] = torch.log(new_scales)
                p_new = p[sampled_idxs]
            else:
                p_new = p[sampled_idxs]
            return torch.nn.Parameter(torch.cat([p, p_new]), requires_grad=p.requires_grad)

        def optimizer_fn(key: str, v: Tensor) -> Tensor:
            v_new = torch.zeros((n, *v.shape[1:]), device=v.device)
            return torch.cat([v, v_new])

        _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)

        # Update state tensors
        for k, v in state.items():
            if isinstance(v, torch.Tensor) and v.shape[0] > 0:
                v_new = torch.zeros((n, *v.shape[1:]), device=v.device)
                state[k] = torch.cat([v, v_new])


class FreeTimeGSCanonicalRunner:
    """Training runner for FreeTimeGS with canonical phase densification."""

    def __init__(
        self,
        local_rank: int,
        world_rank: int,
        world_size: int,
        cfg: FreeTimeCanonicalConfig,
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

        # Create strategy
        if cfg.densification_mode == "default":
            base_strategy = DefaultStrategy(
                prune_opa=cfg.prune_opacity_threshold,
                grow_grad2d=cfg.grow_grad2d_threshold,
                grow_scale3d=cfg.grow_scale3d_threshold,
                refine_start_iter=cfg.densify_start_iter,
                refine_stop_iter=cfg.densify_stop_iter,
                refine_every=cfg.densify_every,
                reset_every=cfg.reset_opacity_every,
                verbose=True,
            )
        else:
            base_strategy = MCMCStrategy(
                cap_max=cfg.max_gaussians,  # Use unified max_gaussians cap
                noise_lr=cfg.mcmc_noise_lr,
                refine_start_iter=cfg.densify_start_iter,
                refine_stop_iter=cfg.densify_stop_iter,
                refine_every=cfg.densify_every,
                min_opacity=cfg.mcmc_min_opacity,
                verbose=True,
            )

        self.strategy = FreeTime4DStrategy(cfg, base_strategy, self.device)
        self.strategy.check_sanity(self.splats, self.optimizers)
        self.strategy_state = self.strategy.initialize_state(scene_scale=self.scene_scale)

        # Run initialization sanity check
        self._init_sanity_check()

        # Losses
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(self.device)

    def _init_sanity_check(self):
        """
        Sanity check for initialization parameters.

        Verifies:
        1. Times are in [0, 1] range
        2. Durations are reasonable (not too small/large)
        3. Velocities are reasonable relative to scene scale
        4. Positions are within scene bounds
        5. Motion model produces reasonable outputs
        """
        cfg = self.cfg
        print(f"\n{'='*70}")
        print("[INITIALIZATION SANITY CHECK]")
        print(f"{'='*70}")

        issues = []
        warnings = []

        with torch.no_grad():
            N = len(self.splats["means"])

            # 1. TIMES CHECK: Should be in [0, 1]
            times = self.splats["times"]
            times_min, times_max = times.min().item(), times.max().item()
            times_mean = times.mean().item()
            print(f"\n[TIMES] (should be in [0, 1]):")
            print(f"  range: [{times_min:.4f}, {times_max:.4f}]")
            print(f"  mean:  {times_mean:.4f}")
            if times_min < -0.1 or times_max > 1.1:
                issues.append(f"TIMES out of range: [{times_min:.4f}, {times_max:.4f}]")
            elif times_min < 0 or times_max > 1:
                warnings.append(f"TIMES slightly out of [0,1]: [{times_min:.4f}, {times_max:.4f}]")

            # 2. DURATIONS CHECK: In log space, exp(durations) = actual spread
            durations = self.splats["durations"]
            dur_min, dur_max = durations.min().item(), durations.max().item()
            dur_actual_min = torch.exp(durations).min().item()
            dur_actual_max = torch.exp(durations).max().item()
            print(f"\n[DURATIONS] (log space, exp() = actual):")
            print(f"  log range:    [{dur_min:.4f}, {dur_max:.4f}]")
            print(f"  actual range: [{dur_actual_min:.4f}, {dur_actual_max:.4f}]")
            print(f"  init_duration (config): {cfg.init_duration}")
            if dur_actual_min < 0.001:
                warnings.append(f"Very small durations detected: exp({dur_min:.2f})={dur_actual_min:.4f}")
            if dur_actual_max > 1e6:
                warnings.append(f"Very large durations detected: exp({dur_max:.2f})={dur_actual_max:.2e}")

            # 3. VELOCITIES CHECK: Should be reasonable relative to scene
            velocities = self.splats["velocities"]
            vel_mag = velocities.norm(dim=-1)
            vel_mean = vel_mag.mean().item()
            vel_max = vel_mag.max().item()
            vel_p90 = torch.quantile(vel_mag, 0.9).item()
            vel_p99 = torch.quantile(vel_mag, 0.99).item()

            # Get position range for context
            means = self.splats["means"]
            pos_range = (means.max() - means.min()).item()

            # Displacement: velocity * (t - μt), where (t - μt) ∈ [-0.5, 0.5]
            max_displacement = vel_max * 0.5
            mean_displacement = vel_mean * 0.5
            displacement_pct = 100 * max_displacement / pos_range if pos_range > 0 else 0

            print(f"\n[VELOCITIES]:")
            print(f"  magnitude: mean={vel_mean:.4f}, p90={vel_p90:.4f}, p99={vel_p99:.4f}, max={vel_max:.4f}")
            print(f"  scene span: {pos_range:.4f} units")
            print(f"  max displacement (vel × 0.5): {max_displacement:.4f} units = {displacement_pct:.1f}% of scene")
            print(f"  mean displacement: {mean_displacement:.4f} units")
            print(f"  velocity_max_magnitude config: {cfg.velocity_max_magnitude}")

            # Warn if displacement is large relative to scene
            if displacement_pct > 20:
                warnings.append(f"Large velocities: max displacement {max_displacement:.2f} = {displacement_pct:.0f}% of scene")
            if vel_max > cfg.velocity_max_magnitude * 5:
                warnings.append(f"Velocity max {vel_max:.2f} >> config cap {cfg.velocity_max_magnitude}")

            # 4. POSITIONS CHECK (means already loaded above)
            pos_min = means.min().item()
            pos_max = means.max().item()
            print(f"\n[POSITIONS]:")
            print(f"  range: [{pos_min:.4f}, {pos_max:.4f}]")
            print(f"  span:  {pos_range:.4f}")
            print(f"  scene_scale: {self.scene_scale:.4f}")

            # 5. TEMPORAL OPACITY TEST: Check at t=0, t=0.5, t=1.0
            print(f"\n[TEMPORAL OPACITY TEST]:")
            for test_t in [0.0, 0.5, 1.0]:
                temp_op = self.compute_temporal_opacity(test_t)
                visible_count = (temp_op > 0.5).sum().item()
                visible_pct = 100 * visible_count / N
                print(f"  t={test_t:.1f}: {visible_count}/{N} ({visible_pct:.1f}%) visible (opacity > 0.5)")
                print(f"        opacity range: [{temp_op.min().item():.4f}, {temp_op.max().item():.4f}]")

            # 6. MOTION MODEL TEST: Check moved positions at t=0 and t=1
            print(f"\n[MOTION MODEL TEST]:")
            for test_t in [0.0, 0.5, 1.0]:
                moved = self.compute_moved_positions(test_t)
                displacement = (moved - means).norm(dim=-1)
                disp_mean = displacement.mean().item()
                disp_max = displacement.max().item()
                print(f"  t={test_t:.1f}: displacement mean={disp_mean:.6f}, max={disp_max:.6f}")

            # 7. LEARNING RATE SANITY CHECK
            print(f"\n[LEARNING RATES]:")
            print(f"  position_lr:    {cfg.position_lr} * scene_scale = {cfg.position_lr * self.scene_scale:.2e}")
            print(f"  velocity_lr:    {cfg.velocity_lr_start} -> {cfg.velocity_lr_end}")
            print(f"  times_lr:       {cfg.times_lr}")
            print(f"  durations_lr:   {cfg.durations_lr}")
            print(f"  scales_lr:      {cfg.scales_lr}")
            print(f"  opacities_lr:   {cfg.opacities_lr}")

        # Print summary
        print(f"\n[SUMMARY]")
        if issues:
            print(f"  CRITICAL ISSUES ({len(issues)}):")
            for issue in issues:
                print(f"    ❌ {issue}")
        if warnings:
            print(f"  WARNINGS ({len(warnings)}):")
            for warning in warnings:
                print(f"    ⚠️  {warning}")
        if not issues and not warnings:
            print(f"  ✓ All checks passed!")

        print(f"{'='*70}\n")

        if issues:
            raise RuntimeError(f"Initialization failed: {issues}")

    def compute_temporal_opacity(self, t: float) -> Tensor:
        """
        Compute temporal opacity σ(t) for all Gaussians at time t.
        σ(t) = exp(-0.5 * ((t - µt) / s)^2)
        """
        mu_t = self.splats["times"]
        s = torch.exp(self.splats["durations"])
        temporal_opacity = torch.exp(-0.5 * ((t - mu_t) / (s + 1e-8)) ** 2)
        return temporal_opacity.squeeze(-1)

    def compute_moved_positions(self, t: float, motion_scale: float = 1.0, use_motion_factor: bool = True) -> Tensor:
        """
        Compute moved positions µx(t) = µx + motion_factor * motion_scale * v * (t - µt)

        GIFStream-style motion gating: Each Gaussian has a learned motion_factor that
        determines how much it should move. This allows the model to learn which
        Gaussians are static vs dynamic.

        Args:
            t: Time at which to compute positions
            motion_scale: Scale factor for velocity contribution (0 to 1).
                         Used for gradual motion warmup during 4D phase.
            use_motion_factor: Whether to apply motion factor gating (default True).
                              Set False during canonical phase to completely disable motion.
        """
        cfg = self.cfg
        mu_x = self.splats["means"]
        mu_t = self.splats["times"]
        v = self.splats["velocities"]
        dt = t - mu_t

        # Motion factor gating (GIFStream-style)
        # effective_velocity = sigmoid(motion_factor) * velocity
        # Motion factor is stored in factors[:, 0:1] (combined factors tensor)
        if use_motion_factor and cfg.use_motion_factor and "factors" in self.splats:
            motion_factor = torch.sigmoid(self.splats["factors"][:, 0:1])  # [N, 1]
            effective_v = motion_factor * v
        else:
            effective_v = v

        # Gradual motion: scale velocity contribution from 0 to 1 during warmup
        moved_positions = mu_x + motion_scale * effective_v * dt
        return moved_positions

    def compute_rotated_quats(self, t: float, motion_scale: float = 1.0) -> Tensor:
        """
        Compute rotated quaternions at time t using angular velocity.

        q(t) = q_base * quaternion_exp(quat_velocity * dt * motion_scale)

        This allows modeling rotating objects (spinning wheels, turning heads, etc.)

        Args:
            t: Time at which to compute quaternions
            motion_scale: Scale factor for angular velocity (0 to 1).
        """
        cfg = self.cfg

        q_base = self.splats["quats"]  # [N, 4]

        # Quat velocity is stored in factors[:, 1:5] (combined factors tensor)
        if not cfg.use_rotation_velocity or "factors" not in self.splats:
            return q_base

        mu_t = self.splats["times"]
        quat_vel = self.splats["factors"][:, 1:5]  # [N, 4]
        dt = (t - mu_t) * motion_scale  # [N, 1]

        # Quaternion exponential: exp(q) where q is pure quaternion (w=0)
        # For small angles: exp([0, v]) ≈ [cos(|v|), sin(|v|) * v/|v|]
        # Here quat_vel represents angular velocity as a quaternion-like vector
        # We scale by dt to get incremental rotation
        scaled_quat_vel = quat_vel * dt  # [N, 4]

        # Compute quaternion exponential
        # Split into scalar and vector parts
        qv_w = scaled_quat_vel[:, 0:1]  # [N, 1]
        qv_xyz = scaled_quat_vel[:, 1:4]  # [N, 3]

        # For rotation: treat as axis-angle and convert to quaternion
        angle = qv_xyz.norm(dim=-1, keepdim=True)  # [N, 1]
        axis = qv_xyz / (angle + 1e-8)  # [N, 3], normalized axis

        # quaternion_exp([0, axis*angle]) = [cos(angle/2), axis*sin(angle/2)]
        half_angle = angle / 2
        exp_w = torch.cos(half_angle)  # [N, 1]
        exp_xyz = axis * torch.sin(half_angle)  # [N, 3]

        # Combine into quaternion
        q_delta = torch.cat([exp_w, exp_xyz], dim=-1)  # [N, 4]

        # Quaternion multiplication: q_final = q_base * q_delta
        # Using Hamilton product: [w1,v1]*[w2,v2] = [w1*w2 - v1·v2, w1*v2 + w2*v1 + v1×v2]
        w1, x1, y1, z1 = q_base[:, 0:1], q_base[:, 1:2], q_base[:, 2:3], q_base[:, 3:4]
        w2, x2, y2, z2 = q_delta[:, 0:1], q_delta[:, 1:2], q_delta[:, 2:3], q_delta[:, 3:4]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        q_rotated = torch.cat([w, x, y, z], dim=-1)  # [N, 4]

        # Normalize to ensure unit quaternion
        q_rotated = F.normalize(q_rotated, dim=-1)

        return q_rotated

    def rasterize_at_time(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        t: float,
        sh_degree: int,
        in_canonical_phase: bool = False,
        in_4d_warmup: bool = False,
        motion_scale: float = 1.0,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        """
        Rasterize Gaussians at a specific time t.

        During canonical phase:
        - Use static positions (no motion): means only
        - Set temporal_opacity = 1.0 (all Gaussians visible for fair pruning)

        During 4D warmup:
        - Use moved positions with gradual motion: µx(t) = µx + motion_scale * v * (t - µt)
        - motion_scale ramps from 0 to 1 during warmup
        - Keep temporal_opacity = 1.0 (adapt to motion first)

        After 4D warmup (full 4D):
        - Use moved positions: µx(t) = µx + v * (t - µt)
        - Compute temporal opacity normally

        Args:
            motion_scale: Scale factor for velocity (0 to 1). Used for gradual motion warmup.
        """
        cfg = self.cfg

        # Compute scales with clamping to prevent invisible Gaussians (black dots)
        scales_raw = torch.exp(self.splats["scales"])
        scales = torch.clamp(scales_raw, min=cfg.scale_min)

        # Minimum opacity to prevent near-invisible Gaussians causing black artifacts
        # Very low opacity + small scale = black dot artifacts
        opacity_min = 1e-4

        if in_canonical_phase:
            # CANONICAL: Train EXACTLY like standard 3DGS (simple_trainer.py)
            # NO temporal modulation - opacity is just sigmoid(opacities)
            #
            # GIFStream-style improvement: If train_velocity_during_canonical is True,
            # we compute positions with motion_scale=0.0. This:
            # - Results in same positions as means (motion_scale=0 → no position change)
            # - BUT velocity is in computation graph → receives gradients
            # - Velocity "learns" motion patterns without affecting rendering
            # - Like GIFStream: motion MLP trains but output is gated to zero
            if cfg.train_velocity_during_canonical:
                # Velocity receives gradients but contributes 0 to position
                # use_motion_factor=False during canonical: motion_factor not applied
                means = self.compute_moved_positions(t, motion_scale=0.0, use_motion_factor=False)
            else:
                # Original: velocity completely excluded from forward pass
                means = self.splats["means"]
            quats = self.splats["quats"]  # No rotation velocity during canonical
            opacities = torch.clamp(torch.sigmoid(self.splats["opacities"]), min=opacity_min)
            temporal_opacity = torch.ones(len(means), device=self.device)  # For info dict only
        elif in_4d_warmup:
            # 4D WARMUP: GRADUAL motion (motion_scale 0→1), temporal_opacity = 1.0
            # This lets model adapt to motion gradually instead of sudden change
            means = self.compute_moved_positions(t, motion_scale=motion_scale, use_motion_factor=True)
            # Gradual rotation: compute rotated quaternions with motion_scale
            quats = self.compute_rotated_quats(t, motion_scale=motion_scale)
            opacities = torch.clamp(torch.sigmoid(self.splats["opacities"]), min=opacity_min)
            temporal_opacity = torch.ones(len(means), device=self.device)  # For info dict only
        else:
            # FULL 4D: motion + temporal opacity modulation + rotation velocity
            means = self.compute_moved_positions(t, motion_scale=1.0, use_motion_factor=True)
            temporal_opacity = self.compute_temporal_opacity(t)
            # Full rotation: compute rotated quaternions
            quats = self.compute_rotated_quats(t, motion_scale=1.0)
            base_opacity = torch.sigmoid(self.splats["opacities"])
            # Modulate opacity with temporal opacity, with minimum to prevent black dots
            opacities = torch.clamp(base_opacity * temporal_opacity, min=opacity_min)

        # Spherical harmonics
        sh0 = self.splats["sh0"]
        shN = self.splats["shN"]
        colors = torch.cat([sh0, shN], dim=1)

        rasterize_mode = "antialiased" if cfg.antialiased else "classic"

        near_plane = kwargs.pop("near_plane", cfg.near_plane)
        far_plane = kwargs.pop("far_plane", cfg.far_plane)

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
            packed=cfg.packed,
            absgrad=isinstance(self.strategy.base_strategy, DefaultStrategy) and
                    hasattr(self.strategy.base_strategy, 'absgrad') and
                    self.strategy.base_strategy.absgrad,
            rasterize_mode=rasterize_mode,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            **kwargs,
        )

        info["temporal_opacity"] = temporal_opacity
        return render_colors, render_alphas, info

    def get_velocity_lr_scale(self, step: int, max_steps: int) -> float:
        """Compute velocity LR scale using exponential annealing: λ(t) = λ0^(1-t) * λ1^t"""
        t = step / max_steps
        lambda_0 = self.cfg.velocity_lr_start
        lambda_1 = self.cfg.velocity_lr_end
        # Exponential interpolation (geometric mean between start and end LR)
        # At t=0: returns 1.0 (use lambda_0)
        # At t=1: returns lambda_1/lambda_0 (decay to lambda_1)
        lr = (lambda_0 ** (1 - t)) * (lambda_1 ** t)
        return lr / lambda_0

    def update_learning_rates(self, step: int, in_warmup: bool, in_canonical: bool, in_transition: bool):
        """Update learning rates based on training phase.

        GIFStream-style improvement: If train_velocity_during_canonical is True,
        velocity is trained during canonical phase (with scaled LR) but its
        contribution to position is zeroed (motion_scale=0). This "pre-trains"
        velocity patterns like GIFStream's motion MLP, so when 4D phase starts,
        velocity already knows approximate motion directions.
        """
        cfg = self.cfg

        if cfg.freeze_temporal_params:
            # User explicitly wants all temporal params frozen
            vel_lr = 0.0
            times_lr = 0.0
            durations_lr = 0.0
        elif in_warmup:
            # Warmup: freeze all temporal params
            vel_lr = 0.0
            times_lr = 0.0
            durations_lr = 0.0
        elif in_canonical or in_transition:
            # GIFStream-style: Train velocity during canonical but zero its contribution
            # This allows velocity to learn motion patterns while positions stay static
            if cfg.train_velocity_during_canonical:
                # Train velocity at reduced LR (output still zeroed via motion_scale=0)
                vel_lr = cfg.velocity_lr_start * cfg.velocity_canonical_lr_scale * math.sqrt(cfg.batch_size)
            else:
                # Original behavior: freeze velocity during canonical
                vel_lr = 0.0
            # Times and durations still frozen during canonical (learned later)
            times_lr = 0.0
            durations_lr = 0.0
        else:
            # Full 4D: annealed velocity LR
            vel_lr_scale = self.get_velocity_lr_scale(step, cfg.max_steps)
            vel_lr = cfg.velocity_lr_start * vel_lr_scale * math.sqrt(cfg.batch_size)
            times_lr = cfg.times_lr * math.sqrt(cfg.batch_size)
            durations_lr = cfg.durations_lr * math.sqrt(cfg.batch_size)

        for pg in self.optimizers["velocities"].param_groups:
            pg["lr"] = vel_lr
        for pg in self.optimizers["times"].param_groups:
            pg["lr"] = times_lr
        for pg in self.optimizers["durations"].param_groups:
            pg["lr"] = durations_lr

        # Combined factors tensor LR (motion_factor + quat_velocity)
        # factors[:, 0] = motion_factor, factors[:, 1:5] = quat_velocity
        if (cfg.use_motion_factor or cfg.use_rotation_velocity) and "factors" in self.optimizers:
            if in_warmup or in_canonical or in_transition:
                # Freeze factors during canonical (all Gaussians contribute equally)
                factors_lr = 0.0
            else:
                # Full 4D: train factors to learn static/dynamic and rotation
                # Use motion_factor_lr since it's the primary factor parameter
                factors_lr = cfg.motion_factor_lr * math.sqrt(cfg.batch_size)
            for pg in self.optimizers["factors"].param_groups:
                pg["lr"] = factors_lr

    def _apply_alternating_freeze(self, step: int) -> str:
        """
        Alternating freeze mechanism for stable 4D training.

        Alternates between:
        - Training velocity while freezing position/opacity
        - Training position/opacity while freezing velocity

        This prevents instability from both position and velocity changing simultaneously,
        since position at time t depends on velocity: µx(t) = µx + v * (t - µt)

        NOTE: Only active after canonical phase (when motion training begins)
        """
        cfg = self.cfg

        # Skip if not enabled
        if not cfg.alternating_freeze_enabled:
            return "BOTH"

        # Resolve auto-start: -1 means start after canonical phase
        alt_start = cfg.alternating_freeze_start
        if alt_start < 0:
            alt_start = cfg.canonical_phase_steps

        # Skip if outside range
        if step < alt_start or step >= cfg.alternating_freeze_stop:
            return "BOTH"

        # Calculate cycle position
        cycle_length = cfg.alternating_velocity_steps + cfg.alternating_appearance_steps
        position_in_cycle = (step - alt_start) % cycle_length

        # Determine phase based on start order
        if cfg.alternating_start_with_position:
            # START WITH POSITION: adapt positions to varying t first
            # Cycle: [0, appearance_steps) = POS, [appearance_steps, cycle_length) = VEL
            in_position_phase = position_in_cycle < cfg.alternating_appearance_steps
        else:
            # START WITH VELOCITY (original behavior)
            # Cycle: [0, velocity_steps) = VEL, [velocity_steps, cycle_length) = POS
            in_position_phase = position_in_cycle >= cfg.alternating_velocity_steps

        if in_position_phase:
            # POSITION PHASE: Train position/opacity, freeze velocity
            phase = "POS"
            # Unfreeze position and opacity
            for pg in self.optimizers["means"].param_groups:
                pg["lr"] = cfg.position_lr * self.scene_scale * math.sqrt(cfg.batch_size)
            for pg in self.optimizers["opacities"].param_groups:
                pg["lr"] = cfg.opacities_lr * math.sqrt(cfg.batch_size)
            for pg in self.optimizers["scales"].param_groups:
                pg["lr"] = cfg.scales_lr * math.sqrt(cfg.batch_size)
            # Freeze velocity
            for pg in self.optimizers["velocities"].param_groups:
                pg["lr"] = 0.0
        else:
            # VELOCITY PHASE: Train velocity, freeze position/opacity
            phase = "VEL"
            # Freeze position and opacity
            for pg in self.optimizers["means"].param_groups:
                pg["lr"] = 0.0
            for pg in self.optimizers["opacities"].param_groups:
                pg["lr"] = 0.0
            for pg in self.optimizers["scales"].param_groups:
                pg["lr"] = 0.0
            # Unfreeze velocity
            vel_lr = cfg.velocity_lr_start * math.sqrt(cfg.batch_size)
            for pg in self.optimizers["velocities"].param_groups:
                pg["lr"] = vel_lr

        return phase

    def _restore_learning_rates(self):
        """Restore all learning rates after alternating freeze period ends."""
        cfg = self.cfg
        BS = cfg.batch_size

        for pg in self.optimizers["means"].param_groups:
            pg["lr"] = cfg.position_lr * self.scene_scale * math.sqrt(BS)
        for pg in self.optimizers["opacities"].param_groups:
            pg["lr"] = cfg.opacities_lr * math.sqrt(BS)
        for pg in self.optimizers["scales"].param_groups:
            pg["lr"] = cfg.scales_lr * math.sqrt(BS)
        for pg in self.optimizers["velocities"].param_groups:
            pg["lr"] = cfg.velocity_lr_start * math.sqrt(BS)

    def _perform_velocity_classification(self):
        """
        Classify Gaussians as static or dynamic based on velocity magnitude.

        At velocity_classification_step:
        1. Compute velocity magnitude for all Gaussians
        2. Keep bottom velocity_static_percentile% as STATIC
        3. For STATIC points:
           - velocity = 0 (no motion)
           - duration = infinity (always visible across all time)
           - Temporal params (times, durations, velocities) are FROZEN
           - Can still train opacity, scale, color
        """
        cfg = self.cfg
        device = self.device

        print(f"\n{'='*70}")
        print(f"[VELOCITY CLASSIFICATION] Step {cfg.velocity_classification_step}")
        print(f"{'='*70}")

        with torch.no_grad():
            velocities = self.splats["velocities"]
            N = velocities.shape[0]

            # Compute velocity magnitude
            vel_mag = velocities.norm(dim=-1)  # [N]

            # Compute threshold for static classification
            static_threshold = torch.quantile(vel_mag, cfg.velocity_static_percentile / 100.0)

            # Classify: below threshold = STATIC
            static_mask = vel_mag <= static_threshold
            dynamic_mask = ~static_mask

            n_static = static_mask.sum().item()
            n_dynamic = dynamic_mask.sum().item()

            print(f"  Total Gaussians: {N}")
            print(f"  Velocity range: [{vel_mag.min():.6f}, {vel_mag.max():.6f}]")
            print(f"  Static threshold (p{cfg.velocity_static_percentile:.0f}): {static_threshold:.6f}")
            print(f"  STATIC: {n_static} ({100*n_static/N:.1f}%)")
            print(f"  DYNAMIC: {n_dynamic} ({100*n_dynamic/N:.1f}%)")

            # Store classification mask for later use (to zero gradients)
            self.static_mask = static_mask.clone()
            self.has_static_classification = True

            # Freeze temporal params for static Gaussians
            if cfg.freeze_static_temporal_params:
                print(f"\n  STATIC points ({n_static}):")
                print(f"    CAN train: position, opacity, scale, color")
                print(f"    CANNOT train: velocity, duration, times (frozen)")
                print(f"    CANNOT: be densified (split/clone)")
                print(f"    -> velocity = 0, duration = inf, times = 0.5")
                print(f"\n  DYNAMIC points ({n_dynamic}):")
                print(f"    CAN train: ALL parameters (full 4D)")
                print(f"    CAN: be densified")

                # Zero out velocities for static Gaussians
                self.splats["velocities"].data[static_mask] = 0.0

                # Set durations to infinity (very large in log space)
                # duration controls temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)^2)
                # With s = exp(13.8) ≈ 1e6, temporal opacity ≈ 1.0 for any t
                self.splats["durations"].data[static_mask] = 13.8  # log(1e6)

                # Set times to 0.5 (middle of sequence) - doesn't matter with inf duration
                self.splats["times"].data[static_mask] = 0.5

        print(f"{'='*70}\n")

        # Log to tensorboard
        if self.world_rank == 0 and hasattr(self, 'writer'):
            self.writer.add_scalar("classification/n_static", n_static, cfg.velocity_classification_step)
            self.writer.add_scalar("classification/n_dynamic", n_dynamic, cfg.velocity_classification_step)
            self.writer.add_scalar("classification/velocity_threshold", static_threshold.item(), cfg.velocity_classification_step)

    def _init_velocity_from_gradients(self):
        """
        Initialize velocity magnitudes based on accumulated position gradients.

        GIFStream-style: High gradient regions likely have motion, so initialize
        them with higher velocity capacity. This gives velocity a "head start"
        rather than learning from scratch.

        Called after canonical phase when transitioning to 4D training.
        """
        cfg = self.cfg
        device = self.device

        print(f"\n{'='*70}")
        print(f"[VELOCITY INIT FROM GRADIENTS] GIFStream-style initialization")
        print(f"{'='*70}")

        with torch.no_grad():
            velocities = self.splats["velocities"]
            means = self.splats["means"]
            N = velocities.shape[0]

            # Get gradient magnitudes from position gradients
            if means.grad is not None:
                grad_mag = means.grad.norm(dim=-1)  # [N]
            else:
                # Fallback: use velocity magnitude from initialization
                print("  WARNING: No position gradients available, skipping gradient-based init")
                return

            # Normalize gradient magnitudes to [0, 1] range
            grad_min = grad_mag.min()
            grad_max = grad_mag.max()
            grad_norm = (grad_mag - grad_min) / (grad_max - grad_min + 1e-8)

            # Scale velocities by gradient magnitude
            # High gradient = likely moving = scale up velocity
            # Low gradient = likely static = scale down velocity
            vel_mag = velocities.norm(dim=-1, keepdim=True)  # [N, 1]
            vel_dir = velocities / (vel_mag + 1e-8)  # [N, 3] normalized direction

            # New magnitude: scale by (0.5 + 0.5 * grad_norm) -> range [0.5, 1.0]
            # This gives high-gradient regions more velocity capacity
            scale_factor = 0.5 + 0.5 * grad_norm.unsqueeze(-1)  # [N, 1]

            # Also boost overall magnitude for high-gradient regions
            # Using GIFStream-style: clamp grad_norm to [0.15, 0.999] and apply inverse sigmoid
            grad_clamped = grad_norm.clamp(0.15, 0.999)
            boost_factor = -torch.log(1.0 / grad_clamped - 1.0)  # Inverse sigmoid
            boost_factor = boost_factor.clamp(-2.0, 2.0)  # Limit extreme values
            boost_factor = torch.sigmoid(boost_factor)  # Back to [0, 1]

            # Apply scaling to velocity magnitude
            new_vel_mag = vel_mag * (0.5 + boost_factor.unsqueeze(-1))
            new_velocities = vel_dir * new_vel_mag

            # Update velocities
            self.splats["velocities"].data = new_velocities

            # Log statistics
            old_mag = vel_mag.squeeze().mean().item()
            new_mag = new_velocities.norm(dim=-1).mean().item()
            print(f"  Total Gaussians: {N}")
            print(f"  Gradient range: [{grad_min:.6f}, {grad_max:.6f}]")
            print(f"  Velocity magnitude: {old_mag:.6f} -> {new_mag:.6f}")
            print(f"  High-gradient (>50%) Gaussians boosted")

        print(f"{'='*70}\n")

        # Mark that initialization has been done
        self.velocity_init_done = True

        # Log to tensorboard
        if self.world_rank == 0 and hasattr(self, 'writer'):
            step = cfg.velocity_init_step if cfg.velocity_init_step > 0 else cfg.canonical_phase_steps
            self.writer.add_scalar("velocity/init_mean_grad", (grad_min + grad_max).item() / 2, step)
            self.writer.add_scalar("velocity/init_mean_vel", new_mag, step)

    def _zero_static_temporal_gradients(self):
        """
        Zero out gradients for temporal parameters of static Gaussians.
        Called after backward() to prevent updates to times/durations/velocities
        for points classified as static.

        NOTE: Static points CAN still train means, opacities, scales, colors.
        Only temporal params are frozen.
        """
        if not getattr(self, 'has_static_classification', False):
            return

        static_mask = self.static_mask

        # Zero gradients for temporal params of static points
        for param_name in ["times", "durations", "velocities"]:
            param = self.splats[param_name]
            if param.grad is not None:
                param.grad.data[static_mask] = 0.0

    def _update_static_mask_after_prune(self, n_after: int):
        """
        Update static_mask after pruning by re-classifying based on velocity.

        gsplat prunes from arbitrary indices, not sequentially from the end.
        Simply truncating the mask would cause misalignment. Instead, we
        re-classify the remaining Gaussians based on their velocity magnitude,
        using the same percentile threshold as the original classification.

        This ensures static_mask is always correct after pruning.
        """
        cfg = self.cfg
        device = self.device

        with torch.no_grad():
            velocities = self.splats["velocities"]
            N = velocities.shape[0]

            assert N == n_after, f"Expected {n_after} Gaussians, got {N}"

            # Re-classify based on velocity magnitude
            vel_mag = velocities.norm(dim=-1)

            # Use same percentile threshold as original classification
            static_threshold = torch.quantile(vel_mag, cfg.velocity_static_percentile / 100.0)

            # Points with velocity = 0 are definitely static (from previous classification)
            # Points with low velocity (below threshold) are static
            static_mask = vel_mag <= static_threshold

            # Update the mask
            self.static_mask = static_mask

            n_static = static_mask.sum().item()
            n_dynamic = (~static_mask).sum().item()
            # print(f"[Prune Update] Static mask updated: {n_static} static, {n_dynamic} dynamic")

    def _mask_static_for_densification(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask out static points from densification by zeroing their gradient info.

        DefaultStrategy uses info["radii"] and accumulated gradients to decide
        which points to split/clone. By zeroing radii for static points,
        they won't be considered for densification.

        Static points:
        - CAN train: means, opacities, scales, colors
        - CANNOT: be densified (split/clone)
        - Temporal params already frozen by _zero_static_temporal_gradients
        """
        if not getattr(self, 'has_static_classification', False):
            return info

        static_mask = self.static_mask

        # Zero out radii for static points - this excludes them from densification
        # DefaultStrategy only densifies points with radii > 0
        # radii shape is typically [1, N, 2] or [N] depending on gsplat version
        if "radii" in info and info["radii"] is not None:
            radii = info["radii"].clone()
            if radii.dim() == 3:
                # Shape [batch, N, 2] - index into the N dimension
                radii[:, static_mask, :] = 0
            elif radii.dim() == 2:
                # Shape [N, 2]
                radii[static_mask, :] = 0
            elif radii.dim() == 1:
                # Shape [N]
                radii[static_mask] = 0
            info["radii"] = radii

        return info

    def _apply_temporal_aware_opacity_boost(self, in_full_4d: bool) -> Optional[Tensor]:
        """
        Compute maximum temporal opacity across multiple time points.

        For temporal-aware pruning: We want to prevent pruning Gaussians that
        are important at ANY time, not just the current sampled time.

        This temporarily boosts opacities before densification based on the
        maximum temporal visibility across sampled times.

        Returns:
            Original opacities tensor (to restore after densification), or None if not applied.
        """
        cfg = self.cfg

        if not in_full_4d or not cfg.temporal_aware_pruning:
            return None

        # Sample multiple time points
        n_times = cfg.temporal_prune_times
        test_times = torch.linspace(0, 1, n_times).tolist()

        # Compute maximum temporal opacity across all test times
        # This ensures Gaussians visible at ANY time are protected from pruning
        max_temporal_opacity = torch.zeros(len(self.splats["means"]), device=self.device)
        for t in test_times:
            temp_op = self.compute_temporal_opacity(t)
            max_temporal_opacity = torch.maximum(max_temporal_opacity, temp_op)

        # Save original opacities
        original_opacities = self.splats["opacities"].data.clone()

        # Compute effective opacity: base_opacity * max_temporal_opacity
        # This "boosts" the opacity for pruning decisions
        base_opacity = torch.sigmoid(self.splats["opacities"])
        effective_opacity = base_opacity * max_temporal_opacity

        # Convert back to logit space for the pruning check
        # sigmoid^-1(x) = log(x / (1 - x))
        # Clamp to avoid numerical issues
        effective_opacity_clamped = effective_opacity.clamp(1e-6, 1 - 1e-6)
        effective_opacity_logit = torch.log(effective_opacity_clamped / (1 - effective_opacity_clamped))

        # Temporarily set opacities to effective values
        self.splats["opacities"].data = effective_opacity_logit.unsqueeze(-1)

        return original_opacities

    def _restore_opacities(self, original_opacities: Optional[Tensor]):
        """Restore original opacities after temporal-aware pruning."""
        if original_opacities is not None:
            self.splats["opacities"].data = original_opacities

    @torch.no_grad()
    def _periodic_relocation(self, step: int, info: Dict[str, Any], current_time: float):
        """
        Periodic relocation from FreeTimeGS paper (arxiv 2506.05348).

        Relocates low-opacity Gaussians to DYNAMIC regions (moving object areas).
        This helps densify moving objects that were missed during initialization.

        Key insight: Static regions are well-covered, moving objects need more Gaussians.
        So we relocate TO dynamic Gaussian locations, not to any high-opacity region.
        """
        cfg = self.cfg

        if not cfg.periodic_relocation_enabled:
            return

        # Only run after static/dynamic classification has happened
        # We need to know which are dynamic to relocate TO dynamic regions
        if not getattr(self, 'has_static_classification', False):
            return

        if step % cfg.periodic_relocation_every != 0:
            return

        # Get current opacity values
        opacities = torch.sigmoid(self.splats["opacities"])

        # Find low-opacity Gaussians (candidates for relocation)
        # Include BOTH static and dynamic - we repurpose unused Gaussians for dynamic regions
        dead_mask = opacities < cfg.periodic_relocation_opacity_threshold

        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return

        # IMPORTANT: Only sample from DYNAMIC Gaussians as source locations
        # This ensures relocated Gaussians ALWAYS go to moving object regions
        # Even if the dead Gaussian was static, it gets moved to dynamic area
        dynamic_mask = ~self.static_mask

        # Also filter out sources with very small scales (would create invisible Gaussians)
        scales_exp = torch.exp(self.splats["scales"]).mean(dim=-1)  # Average scale per Gaussian
        valid_scale_mask = scales_exp > cfg.scale_min * 10  # Must be reasonably sized

        source_candidate_mask = (
            dynamic_mask &
            (~dead_mask) &
            (opacities > cfg.periodic_relocation_opacity_threshold) &
            valid_scale_mask
        )

        n_candidates = source_candidate_mask.sum().item()
        if n_candidates == 0:
            return

        # Compute sampling score from paper: s = λg * ∇g + λo * σ
        # Higher score = region needs more Gaussians
        candidate_opacities = opacities[source_candidate_mask]

        # Get gradient magnitude if available (from densification tracking)
        if "means2d" in info and self.splats["means"].grad is not None:
            grad_mag = self.splats["means"].grad.norm(dim=-1)[source_candidate_mask]
            grad_mag = grad_mag / (grad_mag.max() + 1e-8)  # Normalize
        else:
            grad_mag = torch.zeros_like(candidate_opacities)

        # Sampling score: balance gradient and opacity
        lambda_g, lambda_o = 0.5, 0.5
        sampling_score = lambda_g * grad_mag + lambda_o * candidate_opacities
        probs = sampling_score / sampling_score.sum()
        candidate_indices = torch.where(source_candidate_mask)[0]

        # Sample n_dead sources from candidate (dynamic) Gaussians
        sampled_idx = torch.multinomial(probs, n_dead, replacement=True)
        source_indices = candidate_indices[sampled_idx]

        # Get dead indices
        dead_indices = torch.where(dead_mask)[0]

        # Relocate: copy position, add noise, reset opacity
        noise_scale = 0.1 * torch.exp(self.splats["scales"][source_indices]).mean(dim=-1, keepdim=True)
        noise = torch.randn_like(self.splats["means"][source_indices]) * noise_scale

        # Copy from source to dead
        self.splats["means"].data[dead_indices] = self.splats["means"].data[source_indices] + noise

        # Copy scales but ensure minimum size to prevent invisible Gaussians
        source_scales = self.splats["scales"].data[source_indices]
        min_scale_log = torch.log(torch.tensor(cfg.scale_min * 10, device=self.device))  # Ensure visible
        self.splats["scales"].data[dead_indices] = torch.maximum(source_scales, min_scale_log)

        self.splats["quats"].data[dead_indices] = self.splats["quats"].data[source_indices]

        # IMPORTANT: Reset colors to NEUTRAL GRAY instead of copying from source
        # This prevents black dots from being created by copying dark source colors
        # The model will learn the correct color during training
        # sh0 = 0 → rgb = C0 * 0 + 0.5 = 0.5 (mid-gray)
        self.splats["sh0"].data[dead_indices] = 0.0
        self.splats["shN"].data[dead_indices] = 0.0  # Reset higher-order SH too

        # Copy temporal params smartly to prevent black dots:
        # - Time: set to CURRENT training time (when we detected need for more Gaussians)
        # - Velocity: copy from source (inherit motion direction)
        # - Duration: reset to init_duration (large, visible at ALL times)

        # Set time to current training time - the Gaussian is needed NOW at this time
        self.splats["times"].data[dead_indices] = current_time

        # Copy velocity from source (important: inherit motion from dynamic source)
        self.splats["velocities"].data[dead_indices] = self.splats["velocities"].data[source_indices]

        # Reset duration to LARGE value so Gaussian is visible at ALL times
        # With init_duration=5.0: temporal_opacity ≈ 0.98 even at t=0 or t=1
        # The model will learn appropriate duration during training
        init_duration_log = torch.log(torch.tensor(cfg.init_duration, device=self.device))
        self.splats["durations"].data[dead_indices] = init_duration_log

        # Reset opacity to initial value
        self.splats["opacities"].data[dead_indices] = torch.logit(torch.tensor(cfg.init_opacity, device=self.device))

        # Mark relocated Gaussians as DYNAMIC (they came from dynamic sources)
        if getattr(self, 'has_static_classification', False):
            self.static_mask[dead_indices] = False  # Now dynamic

        # Reset optimizer states for relocated Gaussians
        for name, optimizer in self.optimizers.items():
            for pg in optimizer.param_groups:
                for p in pg["params"]:
                    if p.grad is not None:
                        state = optimizer.state.get(p, {})
                        if "exp_avg" in state:
                            state["exp_avg"][dead_indices] = 0
                        if "exp_avg_sq" in state:
                            state["exp_avg_sq"][dead_indices] = 0

        # DEBUG: Log relocation details to diagnose black dots
        if step % 500 == 0:  # Log more frequently
            n_was_static = dead_mask[self.static_mask].sum().item() if hasattr(self, 'static_mask') else 0

            # Check relocated params
            relocated_opac = torch.sigmoid(self.splats["opacities"][dead_indices]).mean().item()
            relocated_scale = torch.exp(self.splats["scales"][dead_indices]).mean().item()
            relocated_sh0 = self.splats["sh0"].data[dead_indices].mean().item()

            print(f"[Relocation] Step {step}: Relocated {n_dead} Gaussians ({n_was_static} were static)")
            print(f"  Relocated params: opacity={relocated_opac:.3f}, scale={relocated_scale:.6f}, sh0={relocated_sh0:.3f}")
            print(f"  (sh0=0 means gray, trainable color)")

    def _check_black_dot_sources(self, step: int, t: float):
        """Check for Gaussians that might cause black dot artifacts."""
        if step % 1000 != 0:  # Check every 1000 steps
            return

        with torch.no_grad():
            sh0 = self.splats["sh0"].data  # [N, 1, 3]
            scales = torch.exp(self.splats["scales"].data)  # [N, 3]
            opacities = torch.sigmoid(self.splats["opacities"].data)  # [N]

            # Check for very dark colors (sh0 < -1.5 means rgb < 0.08)
            dark_mask = sh0.mean(dim=(1, 2)) < -1.5
            n_dark = dark_mask.sum().item()

            # Check for very small scales
            small_scale_mask = scales.mean(dim=-1) < 1e-4
            n_small = small_scale_mask.sum().item()

            # Check for low opacity + visible scale (potential black dots)
            low_opac_mask = opacities < 0.01
            n_low_opac = low_opac_mask.sum().item()

            # Combined: dark + not-tiny-opacity = potential black dot
            black_dot_risk = dark_mask & (opacities > 1e-4) & (~small_scale_mask)
            n_risky = black_dot_risk.sum().item()

            if n_dark > 0 or n_risky > 0:
                print(f"[BlackDotCheck] Step {step}:")
                print(f"  Dark Gaussians (sh0 < -1.5): {n_dark}")
                print(f"  Small scale (< 1e-4): {n_small}")
                print(f"  Low opacity (< 0.01): {n_low_opac}")
                print(f"  BLACK DOT RISK (dark + visible): {n_risky}")

                # If risky, check if they're static or dynamic
                if n_risky > 0 and hasattr(self, 'static_mask'):
                    risky_static = (black_dot_risk & self.static_mask).sum().item()
                    risky_dynamic = (black_dot_risk & ~self.static_mask).sum().item()
                    print(f"    - {risky_static} are STATIC, {risky_dynamic} are DYNAMIC")

    def _debug_check(self, step: int, loss: torch.Tensor, info: dict = None):
        """
        Comprehensive debug check for training stability.

        Checks for:
        - NaN/Inf in parameters and gradients
        - Extremely large/small values
        - Learning rate issues
        - Gradient explosion/vanishing
        """
        cfg = self.cfg

        if not cfg.debug_enabled:
            return
        if step % cfg.debug_every != 0:
            return

        issues = []
        warnings = []

        # =====================================================================
        # CHECK 1: NaN/Inf in loss
        # =====================================================================
        if cfg.debug_check_nan:
            if torch.isnan(loss) or torch.isinf(loss):
                issues.append(f"[CRITICAL] Loss is NaN/Inf: {loss.item()}")

        # =====================================================================
        # CHECK 2: Parameter values and ranges
        # =====================================================================
        param_stats = {}
        for name, param in self.splats.items():
            data = param.detach()

            # NaN/Inf check
            if cfg.debug_check_nan:
                nan_count = torch.isnan(data).sum().item()
                inf_count = torch.isinf(data).sum().item()
                if nan_count > 0:
                    issues.append(f"[CRITICAL] {name}: {nan_count} NaN values")
                if inf_count > 0:
                    issues.append(f"[CRITICAL] {name}: {inf_count} Inf values")

            # Value range check
            min_val = data.min().item()
            max_val = data.max().item()
            mean_val = data.mean().item()
            std_val = data.std().item()

            param_stats[name] = {
                'min': min_val, 'max': max_val,
                'mean': mean_val, 'std': std_val,
                'shape': list(data.shape)
            }

            # Alert for extreme values
            if abs(max_val) > cfg.debug_alert_threshold or abs(min_val) > cfg.debug_alert_threshold:
                warnings.append(f"[WARNING] {name}: extreme values [{min_val:.2f}, {max_val:.2f}]")

        # =====================================================================
        # CHECK 3: Gradient magnitudes
        # =====================================================================
        grad_stats = {}
        if cfg.debug_check_gradients:
            for name, param in self.splats.items():
                if param.grad is not None:
                    grad = param.grad.detach()

                    # NaN/Inf in gradients
                    if cfg.debug_check_nan:
                        nan_count = torch.isnan(grad).sum().item()
                        inf_count = torch.isinf(grad).sum().item()
                        if nan_count > 0:
                            issues.append(f"[CRITICAL] {name}.grad: {nan_count} NaN values")
                        if inf_count > 0:
                            issues.append(f"[CRITICAL] {name}.grad: {inf_count} Inf values")

                    grad_norm = grad.norm().item()
                    grad_max = grad.abs().max().item()
                    grad_mean = grad.abs().mean().item()

                    grad_stats[name] = {
                        'norm': grad_norm, 'max': grad_max, 'mean': grad_mean
                    }

                    # Gradient explosion
                    if grad_norm > 1000:
                        warnings.append(f"[WARNING] {name}.grad: large gradient norm {grad_norm:.2f}")
                    # Gradient vanishing (but skip if LR is 0)
                    if grad_norm < 1e-10 and grad_norm > 0:
                        warnings.append(f"[WARNING] {name}.grad: vanishing gradient norm {grad_norm:.2e}")

        # =====================================================================
        # CHECK 4: Learning rates
        # =====================================================================
        lr_stats = {}
        if cfg.debug_check_lr:
            for name, optimizer in self.optimizers.items():
                for i, pg in enumerate(optimizer.param_groups):
                    lr = pg.get('lr', 0)
                    lr_stats[name] = lr

        # =====================================================================
        # CHECK 5: Specific 4D parameter checks
        # =====================================================================
        # Velocity magnitude check
        vel_mag = self.splats["velocities"].detach().norm(dim=-1)
        vel_max = vel_mag.max().item()
        vel_mean = vel_mag.mean().item()
        vel_p99 = torch.quantile(vel_mag, 0.99).item() if len(vel_mag) > 0 else 0

        # Expected max displacement = velocity * 0.5 (half time range)
        max_displacement = vel_max * 0.5
        mean_displacement = vel_mean * 0.5

        if vel_max > cfg.velocity_max_magnitude:
            warnings.append(f"[WARNING] velocities: magnitude {vel_max:.2f} > cap {cfg.velocity_max_magnitude}")

        # Analyze velocity gradient contribution
        if self.splats["velocities"].grad is not None:
            vel_grad = self.splats["velocities"].grad.detach()
            vel_grad_norm = vel_grad.norm().item()
            vel_grad_max = vel_grad.abs().max().item()
            vel_lr = lr_stats.get("velocities", 0)

            # Predicted velocity change = grad * lr (simplified, ignoring Adam momentum)
            predicted_vel_change = vel_grad_max * vel_lr
            if predicted_vel_change > 0.1:
                warnings.append(f"[WARNING] Large velocity update: grad*lr = {vel_grad_max:.2e} * {vel_lr:.2e} = {predicted_vel_change:.2e}")

        # Duration check (should be reasonable in log space)
        dur_data = self.splats["durations"].detach()
        dur_min, dur_max = dur_data.min().item(), dur_data.max().item()
        # exp(dur) gives actual duration, reasonable range is [0.01, 100]
        if dur_max > 20:  # exp(20) ≈ 5e8
            warnings.append(f"[WARNING] durations: very large duration (log) {dur_max:.2f}")
        if dur_min < -5:  # exp(-5) ≈ 0.007
            warnings.append(f"[WARNING] durations: very small duration (log) {dur_min:.2f}")

        # Times check (should be in [0, 1])
        times_data = self.splats["times"].detach()
        times_min, times_max = times_data.min().item(), times_data.max().item()
        if times_min < -0.5 or times_max > 1.5:
            warnings.append(f"[WARNING] times: out of expected range [{times_min:.2f}, {times_max:.2f}]")

        # Scales check (log space)
        scales_data = self.splats["scales"].detach()
        scales_min, scales_max = scales_data.min().item(), scales_data.max().item()
        if scales_max > 10:  # exp(10) ≈ 22000
            warnings.append(f"[WARNING] scales: very large scale (log) {scales_max:.2f}")

        # =====================================================================
        # CHECK 6: Freezing mechanism verification
        # =====================================================================
        # Check if alternating freeze is working
        if cfg.debug_check_lr:
            means_lr = lr_stats.get("means", -1)
            vel_lr = lr_stats.get("velocities", -1)

            # During velocity phase: means should have LR=0
            # During position phase: velocities should have LR=0
            # The actual phase is determined by _apply_alternating_freeze,
            # which is called before _debug_check, so we just report current state

            if means_lr == 0 and vel_lr == 0:
                warnings.append("[WARNING] Both means and velocities have LR=0 - nothing training!")
            elif means_lr > 0 and vel_lr > 0:
                # Both training - this is fine during BOTH phase or before alternating
                pass

        # Check static classification gradient zeroing
        if getattr(self, 'has_static_classification', False):
            static_mask = self.static_mask
            n_static = static_mask.sum().item()

            # Verify temporal gradients are zero for static points
            for param_name in ["times", "durations", "velocities"]:
                param = self.splats[param_name]
                if param.grad is not None:
                    static_grad = param.grad.detach()[static_mask]
                    static_grad_norm = static_grad.norm().item()
                    if static_grad_norm > 1e-8:
                        warnings.append(f"[WARNING] {param_name} static grad not zero: {static_grad_norm:.2e}")

        # =====================================================================
        # CHECK 7: Temporal opacity at current time
        # =====================================================================
        # This requires knowing the current time t, which we get from info
        if info is not None and 'temporal_opacity' in info:
            temp_op = info['temporal_opacity']
            visible = (temp_op > 0.5).sum().item()
            total = len(temp_op)
            if visible < total * 0.1:
                warnings.append(f"[WARNING] Only {visible}/{total} ({100*visible/total:.1f}%) Gaussians visible")

        # =====================================================================
        # OUTPUT
        # =====================================================================
        if issues or (warnings and cfg.debug_print_stats):
            print(f"\n{'='*70}")
            print(f"[DEBUG] Step {step}")
            print(f"{'='*70}")

            for issue in issues:
                print(issue)

            for warning in warnings:
                print(warning)

            if cfg.debug_print_stats:
                print(f"\n[PARAM STATS]")
                for name, stats in param_stats.items():
                    print(f"  {name:12s}: min={stats['min']:+.4f} max={stats['max']:+.4f} "
                          f"mean={stats['mean']:+.4f} std={stats['std']:.4f}")

                if grad_stats:
                    print(f"\n[GRAD STATS]")
                    for name, stats in grad_stats.items():
                        print(f"  {name:12s}: norm={stats['norm']:.4e} max={stats['max']:.4e} "
                              f"mean={stats['mean']:.4e}")

                if lr_stats:
                    print(f"\n[LEARNING RATES]")
                    for name, lr in lr_stats.items():
                        print(f"  {name:12s}: {lr:.2e}")

                print(f"\n[4D SPECIFIC]")
                print(f"  velocity_mag: mean={vel_mean:.6f} max={vel_max:.6f}")
                print(f"  durations:    min={dur_min:.2f} max={dur_max:.2f}")
                print(f"  times:        min={times_min:.2f} max={times_max:.2f}")

            print(f"{'='*70}\n")

        # Log to tensorboard
        if self.world_rank == 0:
            for name, stats in param_stats.items():
                self.writer.add_scalar(f"debug/{name}_max", stats['max'], step)
                self.writer.add_scalar(f"debug/{name}_min", stats['min'], step)
            for name, stats in grad_stats.items():
                self.writer.add_scalar(f"debug/{name}_grad_norm", stats['norm'], step)

        # Stop training if critical issues
        if issues:
            print("[DEBUG] CRITICAL ISSUES DETECTED - Consider stopping training!")
            # Optionally raise an exception:
            # raise RuntimeError(f"Training instability detected at step {step}")

    def train(self):
        cfg = self.cfg
        device = self.device
        max_steps = cfg.max_steps

        # Dump config
        if self.world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        # LR schedulers
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

        # Log training phases
        transition_end = cfg.canonical_phase_steps + cfg.transition_phase_steps
        print(f"\n{'='*70}")
        print("[TRAINING PHASES] (GIFStream-style: ALL frames sampled)")
        print(f"  Phase 0 (WARMUP): steps 0 to {cfg.warmup_steps}")
        print(f"    - No densification, velocity FROZEN")
        print(f"    - Basic appearance optimization")
        print(f"  Phase 1 (CANONICAL): steps {cfg.warmup_steps} to {cfg.canonical_phase_steps}")
        print(f"    - ALL frames sampled (t from data), velocity ZEROED (no motion)")
        print(f"    - Densify every {cfg.densify_every} steps (copy time/velocity from parent)")
        print(f"    - temporal_opacity = 1.0 (all visible at all times)")
        print(f"    - Goal: Learn 'consensus' static structure across all times")
        print(f"  Phase 2 (TRANSITION): steps {cfg.canonical_phase_steps} to {transition_end}")
        print(f"    - ALL frames sampled, velocity still FROZEN")
        print(f"    - NO densification")
        four_d_warmup_end = transition_end + cfg.four_d_warmup_steps
        print(f"  Phase 3 (4D WARMUP): steps {transition_end} to {four_d_warmup_end}")
        print(f"    - Time t from data, motion ENABLED, but temporal_opacity = 1.0")
        print(f"    - Adapt to motion before temporal visibility changes")
        print(f"  Phase 4 (FULL 4D): steps {four_d_warmup_end}+")
        print(f"    - Time t from data, FULL temporal opacity, velocity TRAINED")
        print(f"    - Velocity regularization (lambda={cfg.lambda_velocity})")
        print(f"    - Velocity LR annealing: {cfg.velocity_lr_start} -> {cfg.velocity_lr_end}")
        if cfg.alternating_freeze_enabled:
            alt_start = cfg.canonical_phase_steps if cfg.alternating_freeze_start < 0 else cfg.alternating_freeze_start
            print(f"  ALTERNATING TRAINING: steps {alt_start} to {cfg.alternating_freeze_stop}")
            print(f"    - {cfg.alternating_velocity_steps} steps: VELOCITY (positions frozen, no densification)")
            print(f"    - {cfg.alternating_appearance_steps} steps: POSITION (velocity frozen, densification ON)")
        if cfg.velocity_classification_step > 0:
            print(f"  VELOCITY CLASSIFICATION: step {cfg.velocity_classification_step}")
            print(f"    - {cfg.velocity_static_percentile}% lowest velocity -> STATIC")
            print(f"    - Static points: velocity=0, duration=inf, temporal params FROZEN")
            print(f"    - Static points can still update opacity/scale/color")
        print(f"{'='*70}\n")
        print("[FreeTimeGS] Starting training... (first iteration may take 1-2 min for CUDA JIT + LPIPS)")

        # Compute auto velocity classification step if set to -1
        if cfg.velocity_classification_step < 0:
            # Auto: after first position phase in 4D
            # = transition_end + first_phase_steps
            first_phase_steps = cfg.alternating_appearance_steps if cfg.alternating_start_with_position else cfg.alternating_velocity_steps
            auto_classification_step = transition_end + first_phase_steps
            print(f"[FreeTimeGS] Auto velocity classification at step {auto_classification_step}")
        else:
            auto_classification_step = cfg.velocity_classification_step

        # Track velocity init step (after canonical phase ends)
        if cfg.velocity_init_step < 0:
            # Auto: at transition_end (when 4D starts)
            velocity_init_step = transition_end
        else:
            velocity_init_step = cfg.velocity_init_step

        for step in pbar:
            # --- VELOCITY INIT FROM GRADIENTS (GIFStream-style) ---
            # Initialize velocity based on accumulated gradients when transitioning to 4D
            if cfg.init_velocity_from_gradients and step == velocity_init_step and not getattr(self, 'velocity_init_done', False):
                self._init_velocity_from_gradients()

            # --- VELOCITY CLASSIFICATION ---
            if step == auto_classification_step and auto_classification_step > 0 and not hasattr(self, 'classification_done'):
                self._perform_velocity_classification()
                self.classification_done = True

            # Load batch
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0

            # Determine training phase
            if cfg.skip_canonical_phase:
                # FreeTimeGS-style: no canonical phase, train with varying times from start
                in_warmup = step < cfg.warmup_steps
                in_canonical = False
                in_transition = False
                in_full_4d = step >= cfg.warmup_steps
                in_4d_warmup = cfg.warmup_steps <= step < (cfg.warmup_steps + cfg.four_d_warmup_steps)
                four_d_warmup_end = cfg.warmup_steps + cfg.four_d_warmup_steps
            else:
                # Original: canonical phase at fixed time, then transition to 4D
                in_warmup = step < cfg.warmup_steps
                in_canonical = cfg.warmup_steps <= step < cfg.canonical_phase_steps
                in_transition = cfg.canonical_phase_steps <= step < transition_end
                in_full_4d = step >= transition_end
                four_d_warmup_end = transition_end + cfg.four_d_warmup_steps
                in_4d_warmup = transition_end <= step < four_d_warmup_end

            # Compute motion_scale for gradual motion warmup (0 → 1 during 4D warmup)
            # This prevents abrupt position changes when motion is first enabled
            if in_4d_warmup and cfg.four_d_warmup_steps > 0:
                if cfg.skip_canonical_phase:
                    warmup_progress = (step - cfg.warmup_steps) / cfg.four_d_warmup_steps
                else:
                    warmup_progress = (step - transition_end) / cfg.four_d_warmup_steps
                motion_scale = warmup_progress  # Linear ramp 0 → 1
            else:
                motion_scale = 1.0

            # Determine time
            # ALWAYS use sample's actual time - we sample ALL frames
            # During canonical phase, motion is disabled (in_canonical_phase=True)
            # but we still see all time points to learn a "consensus" static representation
            t = data["time"].to(device).mean().item()

            # --- ALTERNATING FREEZE ---
            prev_freeze_phase = getattr(self, '_prev_freeze_phase', None)
            freeze_phase = self._apply_alternating_freeze(step)

            # Log phase transitions
            if freeze_phase != prev_freeze_phase and cfg.alternating_freeze_enabled:
                if freeze_phase == "VEL":
                    print(f"\n--- Step {step}: VELOCITY PHASE (positions frozen, no densification) ---")
                elif freeze_phase == "POS":
                    print(f"\n--- Step {step}: POSITION PHASE (velocity frozen, densification ON) ---")
                self._prev_freeze_phase = freeze_phase

            if step == cfg.alternating_freeze_stop and cfg.alternating_freeze_enabled:
                self._restore_learning_rates()
                print(f"\n--- Alternating training ended at step {step} ---\n")

            # Update learning rates (unless alternating freeze is controlling them)
            if freeze_phase == "BOTH":
                self.update_learning_rates(step, in_warmup, in_canonical, in_transition)

            height, width = pixels.shape[1:3]
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Strategy pre-backward
            # Use static positions (no motion) during warmup, canonical, and transition
            # Only apply motion model in full 4D phase
            use_static = in_warmup or in_canonical or in_transition

            renders, alphas, info = self.rasterize_at_time(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                t=t,
                sh_degree=sh_degree_to_use,
                in_canonical_phase=use_static,  # Static positions when True
                in_4d_warmup=in_4d_warmup,  # Motion + temporal_opacity=1.0
                motion_scale=motion_scale,  # Gradual motion warmup (0→1)
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
            )

            self.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            colors = renders[..., :3]
            colors_clamped = torch.clamp(colors, 0.0, 1.0)

            # Losses
            l1_loss = F.l1_loss(colors_clamped, pixels)
            ssim_loss = 1.0 - fused_ssim(
                colors_clamped.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
                padding="valid"
            )
            lpips_loss = self.lpips(
                colors_clamped.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2)
            )

            loss = (
                cfg.lambda_img * l1_loss +
                cfg.lambda_ssim * ssim_loss +
                cfg.lambda_perc * lpips_loss
            )

            # Regularization losses - only active in FULL 4D phase
            velocity_reg_loss = torch.tensor(0.0, device=device)
            duration_reg_loss = torch.tensor(0.0, device=device)
            temporal_smooth_loss = torch.tensor(0.0, device=device)
            dynamic_opacity_loss = torch.tensor(0.0, device=device)
            reg_4d_loss = torch.tensor(0.0, device=device)

            if in_full_4d:
                # 1. Velocity regularization - penalize large velocities
                if cfg.lambda_velocity > 0:
                    vel_mag = self.splats["velocities"].norm(dim=-1)
                    # Soft penalty: quadratic above velocity_max_magnitude
                    excess_vel = F.relu(vel_mag - cfg.velocity_max_magnitude)
                    velocity_reg_loss = cfg.lambda_velocity * (excess_vel ** 2).mean()
                    loss = loss + velocity_reg_loss

                # 2. Duration regularization - penalize too-short durations
                # Short durations cause Gaussians to appear/disappear quickly
                if cfg.lambda_duration > 0:
                    durations = torch.exp(self.splats["durations"])  # Convert from log
                    # Penalize durations below min threshold
                    short_duration = F.relu(cfg.duration_min - durations)
                    duration_reg_loss = cfg.lambda_duration * (short_duration ** 2).mean()
                    loss = loss + duration_reg_loss

                # 3. Temporal smoothness loss (GIFStream-style)
                # Penalizes large position changes across adjacent time points
                # Encourages smooth motion rather than chaotic velocity patterns
                if cfg.lambda_temporal_smooth > 0:
                    dt = cfg.temporal_smooth_dt
                    # Compute positions at t-dt, t, t+dt
                    pos_t = self.compute_moved_positions(t, motion_scale=1.0)
                    # Only apply smoothness if t-dt and t+dt are in valid range [0, 1]
                    smooth_loss_sum = torch.tensor(0.0, device=device)
                    n_comparisons = 0
                    if t - dt >= 0:
                        pos_prev = self.compute_moved_positions(t - dt, motion_scale=1.0)
                        smooth_loss_sum = smooth_loss_sum + (pos_t - pos_prev).abs().mean()
                        n_comparisons += 1
                    if t + dt <= 1:
                        pos_next = self.compute_moved_positions(t + dt, motion_scale=1.0)
                        smooth_loss_sum = smooth_loss_sum + (pos_t - pos_next).abs().mean()
                        n_comparisons += 1
                    if n_comparisons > 0:
                        temporal_smooth_loss = cfg.lambda_temporal_smooth * smooth_loss_sum / n_comparisons
                        loss = loss + temporal_smooth_loss

                # 4. Dynamic opacity regularization - encourage visible opacity for dynamic
                if cfg.lambda_dynamic_opacity > 0 and hasattr(self, 'static_mask'):
                    dynamic_mask = ~self.static_mask
                    if dynamic_mask.sum() > 0:
                        dynamic_opacity = torch.sigmoid(self.splats["opacities"][dynamic_mask])
                        # Penalize low opacity for dynamic points
                        dynamic_opacity_loss = cfg.lambda_dynamic_opacity * F.relu(0.5 - dynamic_opacity).mean()
                        loss = loss + dynamic_opacity_loss

                # 5. FreeTimeGS 4D Regularization (arxiv 2506.05348)
                # ℒ_reg(t) = (1/N) * Σ(σ * sg[σ(t)])
                # Prevents high-opacity Gaussians from blocking gradient flow
                # Uses stop-gradient on temporal_opacity to only regularize base opacity
                # WARNING: This can hurt dynamic regions - disabled by default
                if cfg.lambda_4d_reg > 0:
                    base_opacity = torch.sigmoid(self.splats["opacities"])
                    temporal_opacity = info.get("temporal_opacity", None)
                    if temporal_opacity is not None:
                        # Stop gradient on temporal opacity - we only want to regularize base opacity
                        reg_4d_loss = cfg.lambda_4d_reg * (base_opacity * temporal_opacity.detach()).mean()
                        loss = loss + reg_4d_loss

                # 6. Motion Factor Sparsity Regularization (GIFStream-style)
                # Encourages binary (0/1) motion factors for clear static/dynamic separation
                # Loss = λ * sigmoid(f) * (1 - sigmoid(f)) which is 0 at 0 and 1, max at 0.5
                # Motion factor is stored in factors[:, 0:1]
                if cfg.lambda_factor_sparsity > 0 and cfg.use_motion_factor and "factors" in self.splats:
                    motion_factor = torch.sigmoid(self.splats["factors"][:, 0:1])  # [N, 1]
                    # Entropy-like loss: f*(1-f) is 0 at extremes, max at 0.5
                    factor_sparsity_loss = cfg.lambda_factor_sparsity * (motion_factor * (1 - motion_factor)).mean()
                    loss = loss + factor_sparsity_loss

                # 7. Quaternion Velocity Regularization
                # Penalize large angular velocities to prevent wild rotations
                # Quat velocity is stored in factors[:, 1:5]
                if cfg.lambda_quat_velocity_reg > 0 and cfg.use_rotation_velocity and "factors" in self.splats:
                    quat_vel = self.splats["factors"][:, 1:5]  # [N, 4]
                    quat_vel_mag = quat_vel.norm(dim=-1)  # [N]
                    quat_velocity_reg_loss = cfg.lambda_quat_velocity_reg * (quat_vel_mag ** 2).mean()
                    loss = loss + quat_velocity_reg_loss

            # Backward
            loss.backward()

            # Update per-frame gradient accumulation (GIFStream-style temporal-aware densification)
            if cfg.use_per_frame_gradients:
                self.strategy.update_per_frame_gradients(
                    state=self.strategy_state,
                    info=info,
                    current_time=t,
                    num_gaussians=len(self.splats["means"]),
                    step=step,
                )

            # Gradient clipping for velocities to prevent explosion
            if self.splats["velocities"].grad is not None and cfg.velocity_grad_clip > 0:
                vel_grad_norm = self.splats["velocities"].grad.norm()
                if vel_grad_norm > cfg.velocity_grad_clip:
                    self.splats["velocities"].grad.mul_(cfg.velocity_grad_clip / vel_grad_norm)

            # Zero gradients for temporal params of static Gaussians
            # (after classification at step 12k, static points keep their temporal values frozen)
            self._zero_static_temporal_gradients()

            # DEBUG: Comprehensive check for training stability
            self._debug_check(step, loss, info)

            # Check for potential black dot sources (only during 4D phase)
            if in_full_4d:
                self._check_black_dot_sources(step, t)

            # Progress bar
            if in_warmup:
                phase_str = "[WARM]"
            elif in_canonical:
                phase_str = "[CANON]"
            elif in_transition:
                phase_str = "[TRANS]"
            elif in_4d_warmup:
                phase_str = f"[4D-W:{motion_scale:.2f}]"  # 4D warmup: motion scaled 0→1
            else:
                phase_str = "[4D]"

            # Add freeze phase indicator
            if freeze_phase == "VEL":
                phase_str += "[V]"
            elif freeze_phase == "POS":
                phase_str += "[P]"

            # Show velocity stats in progress bar
            vel_mag = self.splats["velocities"].detach().norm(dim=-1)
            vel_max = vel_mag.max().item()
            desc = (
                f"{phase_str} loss={loss.item():.4f} | "
                f"l1={l1_loss.item():.4f} | "
                f"t={t:.2f} | "
                f"vel_max={vel_max:.2f} | "
                f"N={len(self.splats['means'])}"
            )
            pbar.set_description(desc)

            # Tensorboard
            if self.world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1_loss", l1_loss.item(), step)
                self.writer.add_scalar("train/ssim_loss", ssim_loss.item(), step)
                self.writer.add_scalar("train/lpips_loss", lpips_loss.item(), step)
                self.writer.add_scalar("train/num_gaussians", len(self.splats["means"]), step)
                self.writer.add_scalar("train/time_t", t, step)
                # Phase: 0=warmup, 1=canonical, 2=transition, 3=4D
                phase_val = 0 if in_warmup else (1 if in_canonical else (2 if in_transition else 3))
                self.writer.add_scalar("train/phase", phase_val, step)
                # Freeze phase: 0=both, 1=velocity, 2=position
                freeze_val = 0 if freeze_phase == "BOTH" else (1 if freeze_phase == "VEL" else 2)
                self.writer.add_scalar("train/freeze_phase", freeze_val, step)
                # Motion scale (for gradual motion warmup)
                self.writer.add_scalar("train/motion_scale", motion_scale, step)

                # Velocity stats - comprehensive monitoring
                vel_mag = self.splats["velocities"].detach().norm(dim=-1)
                self.writer.add_scalar("velocity/mean", vel_mag.mean().item(), step)
                self.writer.add_scalar("velocity/max", vel_mag.max().item(), step)
                self.writer.add_scalar("velocity/std", vel_mag.std().item(), step)
                self.writer.add_scalar("velocity/p90", torch.quantile(vel_mag, 0.9).item(), step)
                self.writer.add_scalar("velocity/p99", torch.quantile(vel_mag, 0.99).item(), step)
                # How many exceed the soft cap
                exceed_cap = (vel_mag > cfg.velocity_max_magnitude).float().mean().item()
                self.writer.add_scalar("velocity/pct_exceed_cap", exceed_cap * 100, step)
                # Velocity regularization loss (if active)
                self.writer.add_scalar("train/velocity_reg_loss", velocity_reg_loss.item(), step)
                # Temporal smoothness loss (GIFStream-style)
                self.writer.add_scalar("train/temporal_smooth_loss", temporal_smooth_loss.item(), step)

                # Velocity gradient stats (if available)
                if self.splats["velocities"].grad is not None:
                    vel_grad_norm = self.splats["velocities"].grad.norm().item()
                    self.writer.add_scalar("velocity/grad_norm", vel_grad_norm, step)

                # Motion factor stats (GIFStream-style velocity gating)
                # Motion factor is stored in factors[:, 0:1]
                if cfg.use_motion_factor and "factors" in self.splats:
                    motion_factor = torch.sigmoid(self.splats["factors"][:, 0:1]).detach()  # [N, 1]
                    self.writer.add_scalar("motion_factor/mean", motion_factor.mean().item(), step)
                    self.writer.add_scalar("motion_factor/std", motion_factor.std().item(), step)
                    # Count of static (< 0.1), dynamic (> 0.9), and in-between
                    n_static = (motion_factor < 0.1).sum().item()
                    n_dynamic = (motion_factor > 0.9).sum().item()
                    n_total = motion_factor.numel()
                    self.writer.add_scalar("motion_factor/pct_static", 100 * n_static / n_total, step)
                    self.writer.add_scalar("motion_factor/pct_dynamic", 100 * n_dynamic / n_total, step)

                # Quaternion velocity stats (rotation)
                # Quat velocity is stored in factors[:, 1:5]
                if cfg.use_rotation_velocity and "factors" in self.splats:
                    quat_vel = self.splats["factors"][:, 1:5].detach()  # [N, 4]
                    quat_vel_mag = quat_vel.norm(dim=-1)  # [N]
                    self.writer.add_scalar("quat_velocity/mean", quat_vel_mag.mean().item(), step)
                    self.writer.add_scalar("quat_velocity/max", quat_vel_mag.max().item(), step)

                # Opacity stats
                base_op = torch.sigmoid(self.splats["opacities"]).detach()
                self.writer.add_scalar("train/base_opacity_mean", base_op.mean().item(), step)

                # Images - GT (left) | Render (right)
                if step % cfg.tb_image_every == 0:
                    # pixels: [B, H, W, 3], colors_clamped: [B, H, W, 3]
                    gt_img = pixels[0]  # [H, W, 3]
                    render_img = colors_clamped[0]  # [H, W, 3]
                    comparison = torch.cat([gt_img, render_img], dim=1)  # [H, 2W, 3]
                    # Convert to [3, H, 2W] for tensorboard
                    comparison_chw = comparison.permute(2, 0, 1).contiguous()
                    self.writer.add_image(
                        "train/gt_vs_render",
                        comparison_chw,
                        step,
                        dataformats='CHW'
                    )
                    # Also log separately for easier comparison
                    self.writer.add_image("images/ground_truth", gt_img.permute(2, 0, 1), step, dataformats='CHW')
                    self.writer.add_image("images/render", render_img.permute(2, 0, 1), step, dataformats='CHW')
                    print(f"[TB] Logged images at step {step}, shape: {comparison_chw.shape}")

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

            # Strategy post-backward (densification)
            # Skip densification during:
            # - Warmup phase (let model stabilize)
            # - Transition phase (let motion stabilize)
            # - Velocity phase (only densify when positions are trainable)
            in_velocity_phase = (freeze_phase == "VEL") and cfg.densify_only_in_position_phase
            skip_densification = in_warmup or in_transition or in_velocity_phase

            # Check if we've exceeded the Gaussian cap
            current_count = len(self.splats["means"])
            if current_count >= cfg.max_gaussians:
                if step % 1000 == 0:  # Log occasionally
                    print(f"[Densification] SKIPPED - at cap ({current_count}/{cfg.max_gaussians})")
                skip_densification = True

            # Option to only densify during canonical phase (prevents 4D explosion)
            if cfg.densify_only_canonical and in_full_4d:
                skip_densification = True

            if not skip_densification:
                n_before = len(self.splats["means"])
                mcmc_lr = schedulers[0].get_last_lr()[0] if isinstance(self.strategy.base_strategy, MCMCStrategy) else None

                # Validate all parameters have consistent sizes before densification
                # This helps catch index mismatch issues early
                param_sizes = {name: param.shape[0] for name, param in self.splats.items()}
                unique_sizes = set(param_sizes.values())
                if len(unique_sizes) > 1:
                    print(f"[WARNING] Parameter size mismatch at step {step}: {param_sizes}")

                # Check for NaN/Inf in critical parameters (can cause CUDA kernel failures)
                for name in ["means", "scales", "opacities"]:
                    if name in self.splats:
                        p = self.splats[name]
                        if torch.isnan(p).any() or torch.isinf(p).any():
                            print(f"[ERROR] {name} has NaN/Inf at step {step}")
                            print(f"  NaN count: {torch.isnan(p).sum().item()}")
                            print(f"  Inf count: {torch.isinf(p).sum().item()}")

                # Mask static points from densification (they can still train position/opacity)
                info_for_densify = self._mask_static_for_densification(info)

                # GIFStream-style: Inject combined temporal gradients for densification
                # This uses gradients accumulated across multiple time points
                if cfg.use_per_frame_gradients:
                    info_for_densify = self.strategy.get_combined_grad_for_densification(
                        state=self.strategy_state,
                        info=info_for_densify,
                    )

                # Temporal-aware pruning: Boost opacity based on max visibility across time
                # This prevents pruning Gaussians that are important at ANY time point
                original_opacities = self._apply_temporal_aware_opacity_boost(in_full_4d)

                self.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info_for_densify,
                    in_canonical_phase=use_static,  # Static positions
                    lr=mcmc_lr,
                )

                # Restore original opacities after densification
                self._restore_opacities(original_opacities)

                n_after = len(self.splats["means"])
                if n_after != n_before:
                    change = n_after - n_before
                    sign = "+" if change > 0 else ""
                    print(f"[Densification] Step {step}: {n_before} -> {n_after} ({sign}{change})")
                    if self.world_rank == 0:
                        self.writer.add_scalar("train/num_gaussians", n_after, step)

                    # Update static_mask if classification has happened
                    # New points from densification are treated as DYNAMIC
                    if getattr(self, 'has_static_classification', False):
                        old_mask = self.static_mask
                        if n_after > n_before:
                            # Points added: new points are DYNAMIC (False)
                            new_entries = torch.zeros(n_after - n_before, dtype=torch.bool, device=device)
                            self.static_mask = torch.cat([old_mask, new_entries])
                        else:
                            # Points pruned: Re-classify based on velocity to get correct mask
                            # gsplat prunes from arbitrary indices, so we can't just truncate
                            # Instead, recompute static/dynamic classification for remaining Gaussians
                            self._update_static_mask_after_prune(n_after)

            # Periodic relocation (FreeTimeGS-style) - relocate low-opacity Gaussians
            # This helps densify moving objects that were missed during initialization
            # Only run in 4D phase when motion is enabled
            if in_full_4d:
                self._periodic_relocation(step, info, current_time=t)

            # Safety check
            if len(self.splats["means"]) == 0:
                print("[ERROR] All Gaussians pruned!")
                break

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
        cfg = self.cfg
        device = self.device

        # Limit evaluation samples for speed
        max_samples = cfg.max_eval_samples if cfg.max_eval_samples > 0 else len(self.valset)
        actual_samples = min(max_samples, len(self.valset))
        print(f"[FreeTimeGS] Running evaluation at step {step} ({actual_samples}/{len(self.valset)} samples)...")

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )

        metrics = defaultdict(list)

        for i, data in enumerate(valloader):
            # Early exit if we've evaluated enough samples
            if cfg.max_eval_samples > 0 and i >= cfg.max_eval_samples:
                break

            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            t = data["time"].to(device).mean().item()
            height, width = pixels.shape[1:3]

            colors, _, _ = self.rasterize_at_time(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                t=t,
                sh_degree=cfg.sh_degree,
                in_canonical_phase=False,
            )

            colors = torch.clamp(colors[..., :3], 0.0, 1.0)

            if self.world_rank == 0:
                # Optionally skip saving rendered images (saves IO time)
                if not cfg.skip_eval_render:
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
            stats["num_gaussians"] = len(self.splats["means"])

            print(
                f"[Eval] PSNR: {stats['psnr']:.3f}, "
                f"SSIM: {stats['ssim']:.4f}, "
                f"LPIPS: {stats['lpips']:.4f}"
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

        data = self.valset[0]
        camtoworld = data["camtoworld"].unsqueeze(0).to(device)
        K = data["K"].unsqueeze(0).to(device)
        height, width = data["image"].shape[:2]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        # Use ffmpeg for MP4 output
        writer = imageio.get_writer(
            f"{video_dir}/trajectory_step{step}.mp4",
            fps=30,
            format='FFMPEG',
            codec='libx264',
            quality=8,
        )

        for i in tqdm.trange(n_frames, desc="Rendering trajectory"):
            t = i / (n_frames - 1)

            colors, _, _ = self.rasterize_at_time(
                camtoworlds=camtoworld,
                Ks=K,
                width=width,
                height=height,
                t=t,
                sh_degree=cfg.sh_degree,
                in_canonical_phase=False,
            )

            colors = torch.clamp(colors[..., :3], 0.0, 1.0)
            frame = (colors.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            writer.append_data(frame)

        writer.close()
        print(f"[FreeTimeGS] Video saved to {video_dir}/trajectory_step{step}.mp4")


def main(local_rank: int, world_rank: int, world_size: int, cfg: FreeTimeCanonicalConfig):
    """Main entry point."""
    runner = FreeTimeGSCanonicalRunner(local_rank, world_rank, world_size, cfg)
    runner.train()
    runner.render_trajectory(step=cfg.max_steps - 1)


if __name__ == "__main__":
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--single-gpu", action="store_true")
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.single_gpu:
        sys.argv = [sys.argv[0]] + remaining

    configs = {
        "default": (
            "FreeTimeGS with canonical phase densification using DefaultStrategy.",
            FreeTimeCanonicalConfig(
                densification_mode="default",
                canonical_phase_steps=5000,
                transition_phase_steps=2000,
                densify_every=100,
                densify_start_iter=500,
                densify_stop_iter=25000,
            ),
        ),
        "fast": (
            "Fast test run with minimal steps.",
            FreeTimeCanonicalConfig(
                max_steps=1000,
                eval_steps=[1000],
                save_steps=[1000],
                warmup_steps=100,  # Short warmup
                canonical_phase_steps=400,  # After warmup
                transition_phase_steps=100,  # Steps 400-500
                densify_start_iter=100,
                densify_stop_iter=800,
                max_init_points=50_000,
                alternating_freeze_start=500,  # Start after transition
                alternating_freeze_stop=1000,
            ),
        ),
        "mcmc": (
            "FreeTimeGS with MCMC-style relocation (no split/clone).",
            FreeTimeCanonicalConfig(
                densification_mode="mcmc",
                canonical_phase_steps=5000,
                transition_phase_steps=2000,
                densify_every=100,
                densify_start_iter=500,
                densify_stop_iter=25000,
                mcmc_cap_max=500_000,
            ),
        ),
        "full": (
            "Full training with longer canonical phase for stability.",
            FreeTimeCanonicalConfig(
                max_steps=50_000,
                eval_steps=[7_000, 15_000, 30_000, 50_000],
                save_steps=[7_000, 15_000, 30_000, 50_000],
                canonical_phase_steps=10000,
                transition_phase_steps=5000,
                densify_stop_iter=40000,
                max_init_points=300_000,
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)

    if pre_args.single_gpu:
        print("[FreeTimeGS] Running in single-GPU mode")
        main(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    else:
        cli(main, cfg, verbose=True)
