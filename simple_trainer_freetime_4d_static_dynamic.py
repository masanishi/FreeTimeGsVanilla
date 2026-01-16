"""
FreeTimeGS: 4D Gaussian Splatting with Static/Dynamic Separation

Extension of FreeTimeGS that separates Gaussians into:
- Static (70%): Zero temporal params, no gradients on temporal params
- Dynamic (30%): Full 4D optimization with velocities

Key Features:
1. Velocity-based classification after warmup phase
2. Gradient hooks to freeze temporal params for static Gaussians
3. Category-aware relocation (static→static, dynamic→dynamic)
4. Label inheritance during densification

Usage:
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d_static_dynamic.py default \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results \
        --start-frame 0 --end-frame 300
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
from typing_extensions import Literal, assert_never

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.FreeTime_dataset import FreeTimeParser, FreeTimeDataset
from utils import knn, rgb_to_sh, set_random_seed


# Label constants
LABEL_STATIC = 0
LABEL_DYNAMIC = 1


@dataclass
class Config:
    """Configuration for FreeTimeGS 4D training with Static/Dynamic separation."""

    # Data paths
    data_dir: str = "data/4d_scene"
    result_dir: str = "results/freetime_4d_static_dynamic"
    windowed_npz_path: Optional[str] = None
    data_factor: int = 1
    test_every: int = 8

    # Frame range
    start_frame: int = 0
    end_frame: int = 300
    frame_step: int = 1

    # Sampling from windowed NPZ
    max_samples: int = 2_000_000
    sample_n_times: int = 10
    sample_high_velocity_ratio: float = 0.8

    # Training
    max_steps: int = 60_000
    batch_size: int = 1
    steps_scaler: float = 1.0
    eval_steps: List[int] = field(default_factory=lambda: [15_000, 30_000, 45_000, 60_000])
    save_steps: List[int] = field(default_factory=lambda: [15_000, 30_000, 45_000, 60_000])
    eval_sample_every: int = 60

    # Model
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opacity: float = 0.5
    init_scale: float = 1.0
    init_duration: float = 0.1  # TUNED: Reduced from 0.2 for sharper temporal profiles

    # Loss weights
    lambda_img: float = 0.8
    lambda_ssim: float = 0.2
    lambda_perc: float = 0.01
    lambda_4d_reg: float = 1e-3  # TUNED: Increased from 1e-4 for better regularization
    lambda_duration_reg: float = 5e-4  # NEW: Penalize wide durations (temporal blur)

    # Training phases
    warmup_steps: int = 500
    canonical_phase_steps: int = 2000  # TUNED: Extended from 500 for better initialization

    # Static/Dynamic Classification
    # Classify after early 4D phase (when velocities have partially converged)
    classify_step: int = 5000  # TUNED: Delayed from 1000 for better velocity estimates
    static_percentile: float = 70.0  # Bottom 70% by velocity are static
    reclassify_every: int = 0  # 0 = only classify once, >0 = reclassify periodically

    # Learning rates
    position_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    quats_lr: float = 1e-3
    opacities_lr: float = 5e-2
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20
    times_lr: float = 5e-4
    durations_lr: float = 1e-3  # TUNED: Increased from 5e-4 for faster duration learning
    velocity_lr_start: float = 5e-3  # TUNED: Increased from 1e-3 (5x) for better motion
    velocity_lr_end: float = 1e-4  # TUNED: Increased from 1e-5 (10x) to maintain learning

    # Periodic relocation
    use_relocation: bool = True
    relocation_every: int = 100
    relocation_start_iter: int = 500
    relocation_stop_iter: int = 50_000
    relocation_opacity_threshold: float = 0.005  # TUNED: Lower threshold (was 0.02) - only relocate truly dead
    relocation_max_ratio: float = 0.01  # Max 1% of Gaussians relocated per step to prevent darkening
    relocation_lambda_grad: float = 0.5
    relocation_lambda_opa: float = 0.5

    # Pruning
    use_pruning: bool = False
    prune_every: int = 500
    prune_start_iter: int = 2000
    prune_stop_iter: int = 20_000
    prune_opacity_threshold: float = 0.005

    # Rendering
    near_plane: float = 0.01
    far_plane: float = 1e10
    packed: bool = False
    antialiased: bool = False

    # Strategy
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=lambda: DefaultStrategy(verbose=True)
    )

    # Misc
    global_scale: float = 1.0
    tb_every: int = 100
    tb_image_every: int = 200
    disable_viewer: bool = True
    lpips_net: Literal["vgg", "alex"] = "alex"

    def adjust_steps(self, factor: float):
        """Scale training steps by factor."""
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)
        self.warmup_steps = int(self.warmup_steps * factor)
        self.canonical_phase_steps = int(self.canonical_phase_steps * factor)
        self.classify_step = int(self.classify_step * factor)
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
    """Load and sample from windowed NPZ file."""
    print(f"\n[WindowedNPZ] Loading: {npz_path}")

    data = np.load(npz_path)
    positions = data['positions'].astype(np.float32)
    velocities = data['velocities'].astype(np.float32)
    colors = data['colors'].astype(np.float32)
    times = data['times'].flatten().astype(np.float32)
    durations = data['durations'].flatten().astype(np.float32) if 'durations' in data else np.ones_like(times) * 0.1

    n_total = len(positions)
    print(f"  Total points: {n_total:,}")
    print(f"  Velocity shape: {velocities.shape}")
    print(f"  Time range: [{times.min():.3f}, {times.max():.3f}]")

    if colors.max() > 1.0:
        colors = colors / 255.0

    if max_samples > 0 and n_total > max_samples:
        print(f"\n  [Sampling] Reducing {n_total:,} to {max_samples:,}")

        unique_times = np.unique(times)
        n_available = len(unique_times)
        actual_n_times = min(n_times, n_available)

        if actual_n_times == n_available:
            selected_times = unique_times
        else:
            indices = np.linspace(0, len(unique_times)-1, actual_n_times, dtype=int)
            selected_times = unique_times[indices]

        print(f"    Sampling from {actual_n_times} times: {selected_times.round(3)}")

        vel_mag = np.linalg.norm(velocities, axis=1)

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

            t_vel = vel_mag[t_indices]
            n_high = min(high_vel_per_time, n_at_time)
            if n_high > 0:
                sorted_idx = np.argsort(t_vel)[::-1]
                high_indices = t_indices[sorted_idx[:n_high]]
                all_indices.extend(high_indices.tolist())

            n_spatial = min(spatial_per_time, n_at_time)
            if n_spatial > 0:
                remaining = np.setdiff1d(t_indices, high_indices if n_high > 0 else np.array([]))
                if len(remaining) > 0:
                    spatial_sample = np.random.choice(remaining, min(n_spatial, len(remaining)), replace=False)
                    all_indices.extend(spatial_sample.tolist())

        all_indices = list(set(all_indices))
        np.random.shuffle(all_indices)
        all_indices = np.array(all_indices[:max_samples])

        positions = positions[all_indices]
        velocities = velocities[all_indices]
        colors = colors[all_indices]
        times = times[all_indices]
        durations = durations[all_indices]

        print(f"    Sampled to {len(positions):,} points")

    if transform is not None:
        R = transform[:3, :3]
        t = transform[:3, 3]
        positions = (positions @ R.T) + t
        velocities = velocities @ R.T

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
    print(f"  Velocity magnitude: [{vel_mags.min():.6f}, {vel_mags.max():.6f}]")

    return {
        'positions': torch.from_numpy(positions),
        'velocities': torch.from_numpy(velocities),
        'colors': torch.from_numpy(colors),
        'times': torch.from_numpy(times).unsqueeze(-1),
        'durations': torch.from_numpy(durations).unsqueeze(-1),
    }


def create_splats_with_optimizers_4d(
    cfg: Config,
    init_data: Dict[str, torch.Tensor],
    scene_scale: float = 1.0,
    device: str = "cuda",
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer], Tensor]:
    """
    Create 4D Gaussian splats with temporal parameters.

    Returns:
        splats: ParameterDict with Gaussian parameters
        optimizers: Dict of optimizers
        labels: Tensor with labels (0=static, 1=dynamic), initialized to all dynamic
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

    min_duration = cfg.init_duration
    if durations.min() > 0:
        durations_clamped = torch.clamp(durations, min=min_duration)
        log_durations = torch.log(durations_clamped)
        print(f"[Init] NPZ durations: [{durations.min():.3f}, {durations.max():.3f}]")
        print(f"[Init] Using clamped durations: [{durations_clamped.min():.3f}, {durations_clamped.max():.3f}]")
    else:
        log_durations = torch.log(torch.full((N, 1), min_duration))
        print(f"[Init] Using config init_duration: {min_duration}")

    sh_colors = torch.zeros((N, (cfg.sh_degree + 1) ** 2, 3))
    sh_colors[:, 0, :] = rgb_to_sh(colors)

    params = [
        ("means", torch.nn.Parameter(points), cfg.position_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), cfg.scales_lr),
        ("quats", torch.nn.Parameter(quats), cfg.quats_lr),
        ("opacities", torch.nn.Parameter(opacities), cfg.opacities_lr),
        ("sh0", torch.nn.Parameter(sh_colors[:, :1, :]), cfg.sh0_lr),
        ("shN", torch.nn.Parameter(sh_colors[:, 1:, :]), cfg.shN_lr),
        ("times", torch.nn.Parameter(times), cfg.times_lr),
        ("durations", torch.nn.Parameter(log_durations), cfg.durations_lr),
        ("velocities", torch.nn.Parameter(velocities), cfg.velocity_lr_start),
    ]

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = cfg.batch_size
    optimizers = {
        name: torch.optim.Adam(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    # Initialize labels: all dynamic initially (will be classified later)
    labels = torch.full((N,), LABEL_DYNAMIC, dtype=torch.long, device=device)

    return splats, optimizers, labels


class FreeTime4DStaticDynamicRunner:
    """FreeTimeGS 4D Gaussian Splatting Trainer with Static/Dynamic Separation."""

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
        print(f"[FreeTime4D-SD] Using {len(test_set)} cameras for validation (every {cfg.test_every}-th of {num_cameras})")

        self.trainset = FreeTimeDataset(self.parser, split="train", test_set=test_set)
        self.valset = FreeTimeDataset(self.parser, split="val", test_set=test_set)
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print(f"[FreeTime4D-SD] Scene scale: {self.scene_scale}")
        print(f"[FreeTime4D-SD] Train: {len(self.trainset)}, Val: {len(self.valset)}")

        # Load windowed NPZ
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

        print(f"[FreeTime4D-SD] After filtering: {len(init_data['positions']):,} Gaussians")

        # Create splats, optimizers, and labels
        self.splats, self.optimizers, self.labels = create_splats_with_optimizers_4d(
            cfg, init_data, self.scene_scale, self.device
        )
        print(f"[FreeTime4D-SD] Initialized {len(self.splats['means']):,} Gaussians")

        # Track if classification has been done
        self.classified = False

        # Strategy state
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(scene_scale=self.scene_scale)
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Gradient accumulator for relocation
        self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)
        self.grad_count = 0

        # Losses
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(self.device)
        else:
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(self.device)

        # Register gradient hooks for temporal params
        self._register_temporal_hooks()

    def _register_temporal_hooks(self):
        """Register gradient hooks to zero out temporal gradients for static Gaussians."""

        def make_hook(param_name: str):
            def hook(grad: Tensor) -> Tensor:
                if not self.classified:
                    return grad  # No masking before classification

                # Create mask: 0 for static, 1 for dynamic
                mask = (self.labels == LABEL_DYNAMIC).float()

                # Expand mask to match gradient shape
                if grad.dim() == 2:
                    mask = mask.unsqueeze(-1).expand_as(grad)
                elif grad.dim() == 1:
                    pass  # Already correct shape

                return grad * mask
            return hook

        # Register hooks on temporal parameters
        for name in ["times", "durations", "velocities"]:
            self.splats[name].register_hook(make_hook(name))

        print("[FreeTime4D-SD] Registered gradient hooks for temporal params")

    def _turbo_colormap(self, values: Tensor) -> Tensor:
        """
        Apply turbo colormap to normalized values [0, 1].

        Args:
            values: [N] tensor with values in [0, 1]
        Returns:
            colors: [N, 3] RGB colors
        """
        x = values.unsqueeze(-1)  # [N, 1]

        # Red channel
        r = (0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943)))))
        # Green channel
        g = (0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604)))))
        # Blue channel
        b = (0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973)))))

        colors = torch.cat([r, g, b], dim=-1).clamp(0, 1)  # [N, 3]
        return colors

    def render_duration_velocity_images(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        t: float,
        static_mode: bool = False,
        use_temporal_opacity: bool = True,
    ) -> Tuple[Tensor, Tensor]:
        """
        Render duration and velocity visualization images using turbo heatmap.

        Returns:
            duration_img: [3, H, W] - duration heatmap (blue=low, red=high)
            velocity_img: [3, H, W] - velocity magnitude heatmap (blue=low, red=high)
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
        durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)  # [N]
        dur_min, dur_max = durations_exp.min(), durations_exp.max()
        dur_normalized = (durations_exp - dur_min) / (dur_max - dur_min + 1e-8)  # [N]

        dur_colors = self._turbo_colormap(dur_normalized)  # [N, 3]

        dur_render, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=dur_colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
            packed=self.cfg.packed,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        duration_img = dur_render[0, ..., :3].clamp(0, 1).permute(2, 0, 1)  # [3, H, W]

        # --- Velocity visualization ---
        vel_mag = self.splats["velocities"].norm(dim=-1)  # [N]
        vel_min, vel_max = vel_mag.min(), vel_mag.max()
        vel_normalized = (vel_mag - vel_min) / (vel_max - vel_min + 1e-8)  # [N]

        vel_colors = self._turbo_colormap(vel_normalized)  # [N, 3]

        vel_render, _, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=vel_colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,
            packed=self.cfg.packed,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
        )
        velocity_img = vel_render[0, ..., :3].clamp(0, 1).permute(2, 0, 1)  # [3, H, W]

        return duration_img, velocity_img

    def classify_static_dynamic(self, step: int):
        """
        Classify Gaussians as static or dynamic based on velocity magnitude.

        Static (bottom static_percentile%): Low velocity, freeze temporal params
        Dynamic (top 100-static_percentile%): High velocity, full optimization
        """
        cfg = self.cfg

        with torch.no_grad():
            # Get velocity magnitudes
            velocities = self.splats["velocities"]
            vel_mag = velocities.norm(dim=-1)

            # Compute threshold at static_percentile
            threshold = torch.quantile(vel_mag, cfg.static_percentile / 100.0)

            # Classify
            static_mask = vel_mag <= threshold
            dynamic_mask = ~static_mask

            # Update labels
            self.labels[static_mask] = LABEL_STATIC
            self.labels[dynamic_mask] = LABEL_DYNAMIC

            n_static = static_mask.sum().item()
            n_dynamic = dynamic_mask.sum().item()
            n_total = len(self.labels)

            print(f"\n[Classification] Step {step}:")
            print(f"  Velocity threshold: {threshold:.6f}")
            print(f"  Static:  {n_static:,} ({100*n_static/n_total:.1f}%)")
            print(f"  Dynamic: {n_dynamic:,} ({100*n_dynamic/n_total:.1f}%)")
            print(f"  Velocity stats - Static: max={vel_mag[static_mask].max():.6f}, "
                  f"Dynamic: min={vel_mag[dynamic_mask].min() if n_dynamic > 0 else 0:.6f}")

            # Zero out temporal params for static Gaussians
            # Velocities: set to zero
            self.splats["velocities"].data[static_mask] = 0.0

            # Durations: set to large value (log(1.0) = 0, exp(2) ≈ 7.4 covers full sequence)
            self.splats["durations"].data[static_mask] = 2.0  # Large duration = visible entire sequence

            # Reset optimizer states for modified static params
            for name in ["times", "durations", "velocities"]:
                opt = self.optimizers[name]
                for group in opt.param_groups:
                    for p in group["params"]:
                        if p in opt.state:
                            state = opt.state[p]
                            # Zero momentum/variance for static Gaussians
                            if "exp_avg" in state:
                                state["exp_avg"][static_mask] = 0
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"][static_mask] = 0

            self.classified = True

            # Log to tensorboard
            self.writer.add_scalar("classification/n_static", n_static, step)
            self.writer.add_scalar("classification/n_dynamic", n_dynamic, step)
            self.writer.add_scalar("classification/velocity_threshold", threshold.item(), step)

    def compute_temporal_opacity(self, t: float) -> Tensor:
        """Temporal opacity: σ(t) = exp(-0.5 * ((t - µt) / s)^2)"""
        mu_t = self.splats["times"]
        s = torch.exp(self.splats["durations"])
        return torch.exp(-0.5 * ((t - mu_t) / (s + 1e-8)) ** 2).squeeze(-1)

    def compute_positions_at_time(self, t: float) -> Tensor:
        """Position at time t: µx(t) = µx + v · (t - µt)"""
        mu_x = self.splats["means"]
        mu_t = self.splats["times"]
        v = self.splats["velocities"]
        return mu_x + v * (t - mu_t)

    def compute_4d_regularization(self, temporal_opacity: Tensor) -> Tensor:
        """4D Regularization: Lreg(t) = (1/N) * Σ(σ * sg[σ(t)])"""
        base_opacity = torch.sigmoid(self.splats["opacities"])
        temporal_opa_sg = temporal_opacity.detach()
        reg = (base_opacity * temporal_opa_sg).mean()
        return reg

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
        """Rasterize 4D Gaussians at time t."""
        if static_mode:
            means = self.splats["means"]
        else:
            means = self.compute_positions_at_time(t)

        if use_temporal_opacity:
            temporal_opacity = self.compute_temporal_opacity(t)
        else:
            temporal_opacity = torch.ones(len(means), device=self.device)

        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        base_opacity = torch.sigmoid(self.splats["opacities"])

        opacities = base_opacity * temporal_opacity
        opacities = torch.clamp(opacities, min=1e-4)

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

    def relocate_gaussians_category_aware(self, step: int) -> Tuple[int, int]:
        """
        Category-aware relocation: static→static, dynamic→dynamic.

        Returns (n_static_relocated, n_dynamic_relocated)
        """
        cfg = self.cfg
        all_relocated_idx = []  # Track all relocated for gradient reset

        with torch.no_grad():
            base_opacity = torch.sigmoid(self.splats["opacities"])
            n_total = len(base_opacity)

            # Find dead Gaussians by category
            dead_mask = base_opacity < cfg.relocation_opacity_threshold

            static_mask = self.labels == LABEL_STATIC
            dynamic_mask = self.labels == LABEL_DYNAMIC

            dead_static = dead_mask & static_mask
            dead_dynamic = dead_mask & dynamic_mask

            n_dead_static = dead_static.sum().item()
            n_dead_dynamic = dead_dynamic.sum().item()

            if n_dead_static == 0 and n_dead_dynamic == 0:
                return 0, 0

            # Cap total relocations to prevent scene darkening
            max_relocate = int(n_total * cfg.relocation_max_ratio)

            # Compute sampling score for alive Gaussians
            if self.grad_count > 0:
                grad_score = self.grad_accum / self.grad_count
                grad_score = grad_score / (grad_score.max() + 1e-8)
            else:
                grad_score = torch.zeros_like(base_opacity)

            def relocate_category(dead_mask_cat: Tensor, alive_mask_cat: Tensor, max_for_cat: int) -> int:
                n_dead_total = dead_mask_cat.sum().item()
                if n_dead_total == 0:
                    return 0

                n_alive = alive_mask_cat.sum().item()
                if n_alive == 0:
                    return 0

                # Cap relocations for this category
                n_to_relocate = min(n_dead_total, max_for_cat)
                if n_to_relocate == 0:
                    return 0

                dead_idx_all = dead_mask_cat.nonzero(as_tuple=True)[0]
                alive_idx = alive_mask_cat.nonzero(as_tuple=True)[0]

                # Pick lowest opacity dead Gaussians if we have more than we can relocate
                if n_dead_total > n_to_relocate:
                    dead_opacities = base_opacity[dead_idx_all]
                    _, sorted_indices = dead_opacities.sort()
                    dead_idx = dead_idx_all[sorted_indices[:n_to_relocate]]
                else:
                    dead_idx = dead_idx_all

                n_dead = len(dead_idx)
                all_relocated_idx.append(dead_idx)

                # Sampling score for alive in this category
                alive_grad = grad_score[alive_idx]
                alive_opa = base_opacity[alive_idx]
                alive_opa_norm = alive_opa / (alive_opa.max() + 1e-8)

                sampling_score = cfg.relocation_lambda_grad * alive_grad + cfg.relocation_lambda_opa * alive_opa_norm
                sampling_score = sampling_score.clamp(min=1e-8)

                probs = sampling_score / sampling_score.sum()
                sampled = torch.multinomial(probs, n_dead, replacement=True)
                source_idx = alive_idx[sampled]

                # Relocate: copy from source to dead
                for name, p in self.splats.items():
                    if name == "means":
                        noise = torch.randn(n_dead, 3, device=self.device) * 0.01 * self.scene_scale
                        p.data[dead_idx] = p.data[source_idx] + noise
                    elif name == "opacities":
                        # Use 80% of source opacity instead of fixed init_opacity
                        source_opa = torch.sigmoid(p.data[source_idx])
                        new_opa = source_opa * 0.8
                        p.data[dead_idx] = torch.logit(new_opa.clamp(0.01, 0.99))
                    else:
                        p.data[dead_idx] = p.data[source_idx]

                    # Reset optimizer state
                    opt = self.optimizers[name]
                    for group in opt.param_groups:
                        for param in group["params"]:
                            if param in opt.state:
                                state = opt.state[param]
                                for key in ["exp_avg", "exp_avg_sq"]:
                                    if key in state:
                                        state[key][dead_idx] = 0

                return n_dead

            # Split max_relocate budget between static and dynamic proportionally
            total_dead = n_dead_static + n_dead_dynamic
            if total_dead > 0:
                max_static = int(max_relocate * n_dead_static / total_dead) + 1
                max_dynamic = int(max_relocate * n_dead_dynamic / total_dead) + 1
            else:
                max_static = max_dynamic = max_relocate // 2

            # Relocate static within static
            alive_static = (~dead_mask) & static_mask
            n_relocated_static = relocate_category(dead_static, alive_static, max_static)

            # Relocate dynamic within dynamic
            alive_dynamic = (~dead_mask) & dynamic_mask
            n_relocated_dynamic = relocate_category(dead_dynamic, alive_dynamic, max_dynamic)

            # Only reset gradient for relocated Gaussians, not all
            if all_relocated_idx:
                all_idx = torch.cat(all_relocated_idx)
                self.grad_accum[all_idx] = 0
            # Don't reset grad_count - keep accumulating

        return n_relocated_static, n_relocated_dynamic

    def handle_densification_labels(self, old_n: int, new_n: int, info: Dict):
        """
        Handle label inheritance when Gaussians are densified.

        When strategy adds new Gaussians via split/clone, the new Gaussians
        should inherit the label of their parent.
        """
        if new_n <= old_n:
            return  # No new Gaussians

        n_new = new_n - old_n

        # For DefaultStrategy split: new Gaussians are appended at the end
        # They inherit from the Gaussian that was split
        # The info dict from gsplat contains 'split_mask' or similar

        # Simple heuristic: new Gaussians inherit label from random alive Gaussians
        # This is a fallback - ideally we'd track parent indices

        with torch.no_grad():
            # Resize labels tensor
            old_labels = self.labels
            self.labels = torch.full((new_n,), LABEL_DYNAMIC, dtype=torch.long, device=self.device)
            self.labels[:old_n] = old_labels

            # New Gaussians: inherit from existing based on sampling
            # Strategy typically clones/splits high-gradient Gaussians
            # For simplicity, mark new ones as dynamic (they're being densified because they're active)
            self.labels[old_n:] = LABEL_DYNAMIC

            print(f"[Densification] Added {n_new} Gaussians, marked as dynamic")

    def handle_pruning_labels(self, keep_mask: Tensor):
        """Handle label updates when Gaussians are pruned."""
        self.labels = self.labels[keep_mask]

    def prune_gaussians(self, step: int) -> int:
        """Prune low-opacity Gaussians."""
        cfg = self.cfg

        with torch.no_grad():
            base_opacity = torch.sigmoid(self.splats["opacities"])
            prune_mask = base_opacity < cfg.prune_opacity_threshold
            n_prune = prune_mask.sum().item()

            if n_prune == 0:
                return 0

            n_remaining = (~prune_mask).sum().item()
            if n_remaining < 1000:
                print(f"[Prune] Would leave only {n_remaining} Gaussians, skipping")
                return 0

            # Update labels before pruning
            keep_mask = ~prune_mask
            self.handle_pruning_labels(keep_mask)

            remove(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                mask=prune_mask,
            )

            self.grad_accum = torch.zeros(len(self.splats["means"]), device=self.device)

        return n_prune

    def train(self):
        cfg = self.cfg
        device = self.device

        if self.world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f, default_flow_style=False)

        max_steps = cfg.max_steps
        canonical_end = cfg.warmup_steps + cfg.canonical_phase_steps

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            )
        ]

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=4, persistent_workers=True, pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        print("\n" + "="*70)
        print("[FreeTime4D-SD] Training Phases with Static/Dynamic Separation:")
        print(f"  Warmup: steps 0-{cfg.warmup_steps}")
        print(f"    - Basic appearance learning")
        print(f"    - All Gaussians treated equally (no classification)")
        print(f"  Canonical: steps {cfg.warmup_steps}-{canonical_end}")
        print(f"    - Static positions, temporal opacity active")
        print(f"  Classification: step {cfg.classify_step}")
        print(f"    - Classify by velocity: {cfg.static_percentile}% static, {100-cfg.static_percentile}% dynamic")
        print(f"    - Static Gaussians: velocities=0, large durations, frozen temporal params")
        print(f"  Full 4D: steps {canonical_end}+")
        print(f"    - Motion enabled for dynamic Gaussians only")
        print(f"    - Static Gaussians: appearance-only optimization")
        print(f"    - Category-aware relocation")
        print("="*70 + "\n")

        global_tic = time.time()
        pbar = tqdm.tqdm(range(max_steps))

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

            static_mode = in_warmup or in_canonical
            use_temporal_opacity = not in_warmup

            # Classification check
            if step == cfg.classify_step and not self.classified:
                self.classify_static_dynamic(step)

            # Optional reclassification
            if cfg.reclassify_every > 0 and step > cfg.classify_step:
                if (step - cfg.classify_step) % cfg.reclassify_every == 0:
                    self.classify_static_dynamic(step)

            # LR adjustment
            if in_warmup:
                for name in ["times", "durations", "velocities"]:
                    for pg in self.optimizers[name].param_groups:
                        pg["lr"] = 0.0
            elif in_canonical:
                for pg in self.optimizers["velocities"].param_groups:
                    pg["lr"] = 0.0
                for pg in self.optimizers["times"].param_groups:
                    pg["lr"] = 0.0
                for pg in self.optimizers["durations"].param_groups:
                    pg["lr"] = cfg.durations_lr * math.sqrt(cfg.batch_size)
            else:
                progress = (step - canonical_end) / max(max_steps - canonical_end, 1)
                vel_lr = cfg.velocity_lr_start * (cfg.velocity_lr_end / cfg.velocity_lr_start) ** progress
                for pg in self.optimizers["velocities"].param_groups:
                    pg["lr"] = vel_lr * math.sqrt(cfg.batch_size)
                for pg in self.optimizers["times"].param_groups:
                    pg["lr"] = cfg.times_lr * math.sqrt(cfg.batch_size)
                for pg in self.optimizers["durations"].param_groups:
                    pg["lr"] = cfg.durations_lr * math.sqrt(cfg.batch_size)

            sh_degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # Track Gaussian count for densification detection
            old_n = len(self.splats["means"])

            # Forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds, Ks, width, height, t, sh_degree,
                static_mode=static_mode, use_temporal_opacity=use_temporal_opacity,
            )
            colors = renders[..., :3]

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # Loss
            colors = torch.clamp(colors, 0.0, 1.0)
            colors_p = colors.permute(0, 3, 1, 2)
            pixels_p = pixels.permute(0, 3, 1, 2)

            l1_loss = F.l1_loss(colors, pixels)
            ssim_val = fused_ssim(colors_p, pixels_p, padding="valid")
            ssim_loss = 1.0 - ssim_val
            lpips_loss = self.lpips(colors_p, pixels_p) if cfg.lambda_perc > 0 else torch.tensor(0.0, device=device)

            reg_4d_loss = torch.tensor(0.0, device=device)
            duration_reg_loss = torch.tensor(0.0, device=device)
            if in_4d and cfg.lambda_4d_reg > 0:
                temporal_opacity = info["temporal_opacity"]
                reg_4d_loss = self.compute_4d_regularization(temporal_opacity)

                # Duration regularization: penalize wide temporal windows (cause blur)
                # Only penalize dynamic Gaussians - static should have large durations
                if self.classified and cfg.lambda_duration_reg > 0:
                    durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)
                    dynamic_mask = self.labels == LABEL_DYNAMIC
                    if dynamic_mask.sum() > 0:
                        # Penalize durations larger than target (0.1)
                        # L_dur = mean(max(0, duration - target)^2)
                        target_duration = cfg.init_duration  # 0.1
                        excess = torch.clamp(durations_exp[dynamic_mask] - target_duration, min=0)
                        duration_reg_loss = (excess ** 2).mean()

            loss_img = cfg.lambda_img * l1_loss
            loss_ssim = cfg.lambda_ssim * ssim_loss
            loss_lpips = cfg.lambda_perc * lpips_loss
            loss_4d_reg = cfg.lambda_4d_reg * reg_4d_loss
            loss_dur_reg = cfg.lambda_duration_reg * duration_reg_loss

            loss = loss_img + loss_ssim + loss_lpips + loss_4d_reg + loss_dur_reg

            loss.backward()

            # Gradient accumulation (only for dynamic Gaussians for relocation score)
            if in_4d and "means" in self.splats and self.splats["means"].grad is not None:
                grad_mag = self.splats["means"].grad.norm(dim=-1)
                if len(self.grad_accum) == len(grad_mag):
                    self.grad_accum += grad_mag
                    self.grad_count += 1

            for opt in self.optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            for sched in schedulers:
                sched.step()

            # Strategy post-backward
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

            # Handle densification (strategy may have added Gaussians)
            new_n = len(self.splats["means"])
            if new_n != old_n:
                self.handle_densification_labels(old_n, new_n, info)
                self.grad_accum = torch.zeros(new_n, device=self.device)
                self.grad_count = 0

            # Pruning
            if cfg.use_pruning:
                if cfg.prune_start_iter <= step < cfg.prune_stop_iter:
                    if step % cfg.prune_every == 0:
                        n_pruned = self.prune_gaussians(step)
                        if n_pruned > 0:
                            print(f"[Prune] Step {step}: removed {n_pruned}, remaining {len(self.splats['means'])}")

            # Category-aware relocation
            if cfg.use_relocation and in_4d and self.classified:
                if cfg.relocation_start_iter <= step < cfg.relocation_stop_iter:
                    if step % cfg.relocation_every == 0:
                        n_static, n_dynamic = self.relocate_gaussians_category_aware(step)
                        if n_static > 0 or n_dynamic > 0:
                            print(f"[Relocation] Step {step}: static={n_static}, dynamic={n_dynamic}")

            # Progress bar
            phase = "WARM" if in_warmup else ("CANON" if in_canonical else "4D")
            n_static = (self.labels == LABEL_STATIC).sum().item() if self.classified else 0
            n_dynamic = (self.labels == LABEL_DYNAMIC).sum().item() if self.classified else len(self.labels)
            pbar.set_description(
                f"[{phase}] loss={loss.item():.4f} N={len(self.splats['means'])} "
                f"S={n_static} D={n_dynamic}"
            )

            # Tensorboard logging
            if self.world_rank == 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3

                self.writer.add_scalar("loss/total", loss.item(), step)
                self.writer.add_scalar("loss/l1_raw", l1_loss.item(), step)
                self.writer.add_scalar("loss/ssim_raw", ssim_loss.item(), step)
                self.writer.add_scalar("loss/lpips_raw", lpips_loss.item(), step)
                self.writer.add_scalar("loss/4d_reg_raw", reg_4d_loss.item(), step)
                self.writer.add_scalar("loss/duration_reg_raw", duration_reg_loss.item(), step)

                self.writer.add_scalar("gaussians/count", len(self.splats["means"]), step)
                self.writer.add_scalar("gaussians/static", n_static, step)
                self.writer.add_scalar("gaussians/dynamic", n_dynamic, step)
                self.writer.add_scalar("gaussians/mem_gb", mem, step)

                with torch.no_grad():
                    base_opacity = torch.sigmoid(self.splats["opacities"])
                    self.writer.add_scalar("gaussians/opacity_mean", base_opacity.mean().item(), step)

                    vel_mag = self.splats["velocities"].norm(dim=-1)
                    durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)

                    self.writer.add_scalar("temporal/velocity_mean", vel_mag.mean().item(), step)
                    self.writer.add_scalar("temporal/velocity_max", vel_mag.max().item(), step)
                    self.writer.add_scalar("temporal/duration_mean", durations_exp.mean().item(), step)
                    self.writer.add_scalar("temporal/duration_min", durations_exp.min().item(), step)

                    if self.classified:
                        static_mask = self.labels == LABEL_STATIC
                        dynamic_mask = self.labels == LABEL_DYNAMIC

                        # Dynamic region metrics (hands, face, etc.)
                        if dynamic_mask.sum() > 0:
                            self.writer.add_scalar("dynamic/velocity_mean", vel_mag[dynamic_mask].mean().item(), step)
                            self.writer.add_scalar("dynamic/velocity_max", vel_mag[dynamic_mask].max().item(), step)
                            self.writer.add_scalar("dynamic/duration_mean", durations_exp[dynamic_mask].mean().item(), step)
                            self.writer.add_scalar("dynamic/opacity_mean", base_opacity[dynamic_mask].mean().item(), step)

                        # Static region metrics (background)
                        if static_mask.sum() > 0:
                            self.writer.add_scalar("static/velocity_mean", vel_mag[static_mask].mean().item(), step)
                            self.writer.add_scalar("static/duration_mean", durations_exp[static_mask].mean().item(), step)
                            self.writer.add_scalar("static/opacity_mean", base_opacity[static_mask].mean().item(), step)

                    # Learning rate tracking
                    self.writer.add_scalar("train/lr_velocity", self.optimizers["velocities"].param_groups[0]["lr"], step)
                    self.writer.add_scalar("train/lr_duration", self.optimizers["durations"].param_groups[0]["lr"], step)

                phase_num = 0 if in_warmup else (1 if in_canonical else 2)
                self.writer.add_scalar("train/phase", phase_num, step)

            if self.world_rank == 0 and step % cfg.tb_image_every == 0:
                with torch.no_grad():
                    gt_img = pixels[0].clamp(0, 1)
                    render_img = colors[0].detach().clamp(0, 1)
                    comparison = torch.cat([gt_img, render_img], dim=1)
                    comparison = comparison.permute(2, 0, 1)
                    self.writer.add_image("images/gt_vs_render", comparison, step)

            # --- Duration & Velocity Visualization (every 500 steps) ---
            if self.world_rank == 0 and step % 500 == 0 and step > 0:
                with torch.no_grad():
                    duration_img, velocity_img = self.render_duration_velocity_images(
                        camtoworlds, Ks, width, height, t,
                        static_mode=static_mode, use_temporal_opacity=use_temporal_opacity,
                    )
                    # duration_img: [3, H, W] - turbo heatmap of durations
                    # velocity_img: [3, H, W] - turbo heatmap of velocity magnitudes
                    self.writer.add_image("visualization/duration", duration_img, step)
                    self.writer.add_image("visualization/velocity", velocity_img, step)

                    # Log the min/max values for reference
                    durations_exp = torch.exp(self.splats["durations"]).squeeze(-1)
                    vel_mag = self.splats["velocities"].norm(dim=-1)
                    print(f"[Vis] Step {step}: duration=[{durations_exp.min():.4f}, {durations_exp.max():.4f}], "
                          f"velocity=[{vel_mag.min():.4f}, {vel_mag.max():.4f}]")

            if self.world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.flush()

            # Save checkpoint
            if step in [s - 1 for s in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                    "num_static": n_static,
                    "num_dynamic": n_dynamic,
                }
                print(f"[Save] Step {step}: {stats}")
                with open(f"{self.stats_dir}/train_step{step:04d}.json", "w") as f:
                    json.dump(stats, f)

                data = {
                    "step": step,
                    "splats": self.splats.state_dict(),
                    "labels": self.labels.cpu(),
                    "classified": self.classified,
                }
                torch.save(data, f"{self.ckpt_dir}/ckpt_{step}.pt")

            # Evaluation
            if step in [s - 1 for s in cfg.eval_steps]:
                self.eval(step)

        print(f"\n[Training] Complete! Total time: {time.time() - global_tic:.1f}s")

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        """Evaluate on validation set."""
        cfg = self.cfg
        device = self.device

        valloader = torch.utils.data.DataLoader(self.valset, batch_size=1, shuffle=False)
        metrics = defaultdict(list)
        ellipse_time = 0
        eval_count = 0

        sample_every = cfg.eval_sample_every
        total_frames = len(valloader)
        eval_frames = total_frames // sample_every
        print(f"\n[Eval] Step {step}: evaluating {eval_frames} frames")

        for i, data in enumerate(valloader):
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

            if self.world_rank == 0:
                canvas = torch.cat([pixels, colors], dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(f"{self.render_dir}/{stage}_step{step}_{i:04d}.png", canvas)

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
                f"Time: {ellipse_time:.3f}s/img"
            )
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()


def main(local_rank: int, world_rank: int, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = FreeTime4DStaticDynamicRunner(local_rank, world_rank, world_size, cfg)
    runner.train()


if __name__ == "__main__":
    """
    Usage:

    # Training with DefaultStrategy
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d_static_dynamic.py default \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results \
        --start-frame 0 --end-frame 300

    # Training with MCMCStrategy
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d_static_dynamic.py mcmc \
        --data-dir /path/to/data \
        --windowed-npz-path /path/to/windowed_points.npz \
        --result-dir /path/to/results

    # Adjust static/dynamic ratio
    CUDA_VISIBLE_DEVICES=0 python simple_trainer_freetime_4d_static_dynamic.py default \
        --static-percentile 80.0 \
        ...
    """

    configs = {
        "default": (
            "FreeTimeGS 4D with Static/Dynamic separation using DefaultStrategy.",
            Config(
                strategy=DefaultStrategy(
                    verbose=True,
                    refine_start_iter=500,
                    refine_stop_iter=40_000,
                    reset_every=6000,
                    refine_every=100,
                    prune_opa=0.005,
                    grow_grad2d=0.0002,
                ),
            ),
        ),
        "mcmc": (
            "FreeTimeGS 4D with Static/Dynamic separation using MCMCStrategy.",
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
                    cap_max=4_000_000,
                ),
            ),
        ),
    }

    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    cli(main, cfg, verbose=True)
