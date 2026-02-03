# FreeTimeGsVanilla: Copilot instructions

## 応答言語
- すべての回答・説明は日本語で行う。

## Big picture
- This repo implements FreeTimeGS (4D Gaussian Splatting) on top of `gsplat`.
- Core pipeline is **keyframe combine → 4D training**:
  - `src/combine_frames_fast_keyframes.py` merges per-frame point clouds, estimates velocity, and writes an **NPZ**.
  - `src/simple_trainer_freetime_4d_pure_relocation.py` trains 4D Gaussians from the NPZ plus COLMAP data.
- Viewer for trained checkpoints lives in `src/viewer_4d.py`.

## Data flow / inputs
- Per-frame inputs: `points3d_frameXXXXXX.npy` + `colors_frameXXXXXX.npy` (see README).
- COLMAP sparse model expected at `data_dir/sparse/0/{cameras.bin,images.bin,points3D.bin}` and images under `data_dir/images/`.
- NPZ output schema described in `README.md` (positions/velocities/colors/times/durations/has_velocity).

## Key scripts & workflows
- Full pipeline: `run_pipeline.sh` (activates `.venv`, builds NPZ, then trains).
- Example training shortcuts: `run_small.sh` / `run_full.sh` (hardcoded paths; treat as templates).

## Conventions & patterns
- Use Python **3.12+** (see `pyproject.toml`).
- External deps include `torch`, `pycolmap`, `gsplat` (git source), `viser`, `nerfview`, `open3d`.
- Config name for trainer is a positional arg: `default_keyframe` (≈15M) or `default_keyframe_small` (≈4M).
- For experiments, avoid over-splitting one-off logic into many tiny functions; keep single-use flows cohesive.

## Practical hints for agents
- Respect the frame range semantics: `run_pipeline.sh` passes `--frame-end END-1` to the combiner and `END` to the trainer.
- Many commands rely on `CUDA_VISIBLE_DEVICES` and GPU IDs; keep that in examples.
- If you add new scripts, align with existing Bash style and print clear usage.
