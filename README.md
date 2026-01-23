
# FreeTimeGSVanilla

### Gsplat-based 4D Gaussian Splatting for Dynamic Scenes

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)

A vanilla minimal implementation of **FreeTimeGS** built on [gsplat](https://github.com/nerfstudio-project/gsplat) for reconstructing dynamic scenes from multi-view video.



**Key Features**

- **4D Gaussian Primitives** - Each Gaussian has position, velocity, time, and duration
- **Temporal Motion Model** - `x(t) = x + v * (t - t_canonical)`
- **gsplat Backend** - Efficient CUDA kernels for fast rendering
- **Flexible Optimization** - MCMC and DefaultStrategy densification
- **Keyframe Processing** - Smart sampling for large video sequences


**Based on the paper:** *FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction* Yifan Wang, Peishan Yang, Zhen Xu, Jiaming Sun, Zhanhua Zhang, Yong Chen, Hujun Bao, Sida Peng, Xiaowei Zhou **CVPR 2025** [[Paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.pdf) [[Project Page]](https://zju3dv.github.io/freetimegs/)


---


## Repository Structure

```
FreeTimeGsVanilla/
│
├── src/                          # Core source code
│   ├── simple_trainer_freetime_4d_pure_relocation.py   # Main 4D GS trainer
│   ├── combine_frames_fast_keyframes.py                # Keyframe point cloud combiner
│   └── utils.py                                        # Utility functions (KNN, colormap, etc.)
│
├── datasets/                     # Data loading & processing
│   ├── __init__.py               # Package exports
│   ├── FreeTime_dataset.py       # Dataset class (COLMAP poses, images)
│   ├── normalize.py              # Scene normalization utilities
│   ├── traj.py                   # Camera trajectory generation
│   └── read_write_model.py       # COLMAP binary/text I/O
│
├── run_pipeline.sh               # Full pipeline (combine + train)
├── run_small.sh                  # Quick training (4M points)
├── run_full.sh                   # Full training (15M points)
│
├── LICENSE                       # AGPL-3.0 license
└── README.md                     # This file
```

## Pipeline Overview

The training pipeline consists of two main steps:

1. **Point Cloud Preparation** (`src/combine_frames_fast_keyframes.py`):
   - Loads per-frame triangulated 3D points
   - Extracts keyframes at specified intervals
   - Estimates velocity using k-NN matching between consecutive keyframes
   - Outputs an NPZ file with positions, velocities, colors, and timestamps

2. **4D Gaussian Training** (`src/simple_trainer_freetime_4d_pure_relocation.py`):
   - Initializes 4D Gaussians from the NPZ file
   - Trains with temporal parameters (position, velocity, time, duration)
   - Outputs PLY sequences and trajectory videos

## Keyframes vs All Frames (Stride/Step)

### Why Keyframes?

Processing every single frame of a video is computationally expensive and often redundant. Adjacent frames are typically very similar. Instead, we use **keyframes** - frames sampled at regular intervals.

### Keyframe Step (Stride)

The `--keyframe-step` parameter controls how many frames to skip between keyframes:

- **Step = 1**: Use ALL frames (no skipping) - most accurate but slowest
- **Step = 5**: Use every 5th frame (0, 5, 10, 15, ...) - good balance
- **Step = 10**: Use every 10th frame - faster but less temporal detail

**Example**: For a 60-frame video with `--keyframe-step 5`:
```
Frames:    0  1  2  3  4  5  6  7  8  9  10 11 12 ... 55 56 57 58 59
Keyframes: *              *              *              *
           0              5              10             55
```

This extracts 12 keyframes instead of 60 frames, reducing memory and computation by ~5x while preserving motion information.

### Velocity Estimation

Velocity is computed between consecutive **keyframes** (not all frames):
```
v = (position_keyframe[t+step] - position_keyframe[t]) / step
```

This gives the average velocity over the keyframe interval.

## NPZ File Format

The NPZ file contains the initial 4D Gaussian data:

| Field | Shape | Description |
|-------|-------|-------------|
| `positions` | [N, 3] | 3D coordinates (x, y, z) |
| `velocities` | [N, 3] | Velocity vectors (vx, vy, vz) |
| `colors` | [N, 3] | RGB colors normalized to [0, 1] |
| `times` | [N, 1] | Normalized timestamps in [0, 1] |
| `durations` | [N, 1] | Temporal duration (visibility window) |
| `has_velocity` | [N] | Boolean mask for valid velocity estimates |

**Metadata fields:**
- `frame_start`, `frame_end`: Frame range
- `n_keyframes`: Number of keyframes used
- `keyframe_step`: Step between keyframes
- `mode`: Processing mode identifier

### Example NPZ Creation

```python
import numpy as np

# Your triangulated point clouds (one per frame)
points_frame_0 = np.load("points3d_frame000000.npy")  # [M, 3]
colors_frame_0 = np.load("colors_frame000000.npy")    # [M, 3], values 0-255

# Combine and save
np.savez(
    "init_points.npz",
    positions=positions,      # [N, 3] float32
    velocities=velocities,    # [N, 3] float32
    colors=colors / 255.0,    # [N, 3] float32, normalized to [0, 1]
    times=times,              # [N, 1] float32, normalized to [0, 1]
    durations=durations,      # [N, 1] float32
    has_velocity=has_velocity # [N] bool
)
```

## Input Requirements

### Per-Frame Point Cloud Files

The `src/combine_frames_fast_keyframes.py` script expects:

```
input_dir/
├── points3d_frame000000.npy   # [M, 3] float32 - 3D positions
├── colors_frame000000.npy     # [M, 3] float32 - RGB colors (0-255)
├── points3d_frame000001.npy
├── colors_frame000001.npy
├── ...
└── points3d_frameXXXXXX.npy
```

These are typically generated by triangulating matched features across camera views.

### COLMAP Data

The trainer expects a COLMAP sparse reconstruction:

```
data_dir/
├── images/                    # Or images_Nx/ for downsampled
│   ├── cam01_frame000000.jpg
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

## Usage

### Full Pipeline

```bash
bash run_pipeline.sh \
    /path/to/triangulation/output \   # Input: per-frame NPY files
    /path/to/colmap/data \            # COLMAP reconstruction
    /path/to/results \                # Output directory
    0 \                               # Start frame
    61 \                              # End frame
    5 \                               # Keyframe step
    0 \                               # GPU ID
    default_keyframe_small            # Config name
```

### Step by Step

**Step 1: Combine keyframes**

```bash
python src/combine_frames_fast_keyframes.py \
    --input-dir /path/to/triangulation/output \
    --output-path /path/to/keyframes.npz \
    --frame-start 0 \
    --frame-end 60 \
    --keyframe-step 5
```

**Step 2: Train 4D Gaussians**

```bash
CUDA_VISIBLE_DEVICES=0 python src/simple_trainer_freetime_4d_pure_relocation.py default_keyframe \
    --data-dir /path/to/colmap/data \
    --init-npz-path /path/to/keyframes.npz \
    --result-dir /path/to/results \
    --start-frame 0 \
    --end-frame 61 \
    --max-steps 30000
```

### Available Configs

| Config | Points | Description |
|--------|--------|-------------|
| `default_keyframe` | ~15M | Full resolution, higher quality |
| `default_keyframe_small` | ~4M | Reduced points, faster training |

## Outputs

After training, you'll find:

```
results/
├── ckpts/
│   └── ckpt_30000.pt              # Model checkpoint
├── videos/
│   ├── traj_4d_step30000.mp4      # RGB trajectory video
│   ├── traj_duration_step30000.mp4    # Duration heatmap
│   └── traj_velocity_step30000.mp4    # Velocity heatmap
├── ply_sequence_step30000/
│   ├── frame_000000.ply           # Per-frame PLY exports
│   └── ...
└── tb/                            # TensorBoard logs
```

## Key Parameters

### Point Cloud Preparation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--keyframe-step` | 5 | Frames between keyframes |
| `--max-velocity-distance` | 0.5 | Max k-NN match distance |
| `--sample-ratio` | 1.0 | Point subsampling ratio |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-steps` | 60000 | Training iterations |
| `--init-duration` | 0.1 | Initial temporal duration |
| `--velocity-lr-start` | 5e-3 | Initial velocity learning rate |
| `--velocity-lr-end` | 1e-4 | Final velocity learning rate |
| `--lambda-4d-reg` | 1e-3 | 4D regularization weight |

## 4D Gaussian Parameters

Each Gaussian has 8 learnable parameter groups:

1. **Position (x)**: [N, 3] - Canonical 3D position
2. **Time (t)**: [N, 1] - When the Gaussian is most visible
3. **Duration (s)**: [N, 1] - Temporal width
4. **Velocity (v)**: [N, 3] - Linear velocity
5. **Scale**: [N, 3] - 3D scale
6. **Quaternion**: [N, 4] - Rotation
7. **Opacity**: [N] - Base opacity
8. **Spherical Harmonics**: [N, K, 3] - View-dependent color

### Motion Model

Position at time t:
```
x(t) = x + v * (t - t_canonical)
```

Temporal opacity (Gaussian falloff):
```
opacity(t) = exp(-0.5 * ((t - t_canonical) / duration)^2)
```
## Citation

If you find this work useful, please cite the original paper:

```bibtex
@InProceedings{Wang_2025_CVPR,
    author    = {Wang, Yifan and Yang, Peishan and Xu, Zhen and Sun, Jiaming and Zhang, Zhanhua and Chen, Yong and Bao, Hujun and Peng, Sida and Zhou, Xiaowei},
    title     = {FreeTimeGS: Free Gaussian Primitives at Anytime Anywhere for Dynamic Scene Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {21750-21760}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
