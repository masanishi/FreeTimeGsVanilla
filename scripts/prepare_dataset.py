#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import subprocess


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def run_ffmpeg_extract(
    video_path: Path,
    images_dir: Path,
    cam_name: str,
    frame_start: int,
    frame_end: int,
    fps: int,
    downsample: int,
) -> None:
    if frame_end < frame_start:
        raise ValueError("frame_end は frame_start 以上で指定してください")

    start_time = frame_start / fps
    duration = (frame_end - frame_start + 1) / fps

    vf_filter = f"fps={fps}"
    if downsample != 1:
        vf_filter += f",scale=iw/{downsample}:ih/{downsample}"

    output_pattern = images_dir / f"{cam_name}_frame%06d.png"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.6f}",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        vf_filter,
        "-vsync",
        "0",
        "-start_number",
        "0",
        str(output_pattern),
    ]

    subprocess.run(cmd, check=True)


def _configure_extraction_options(pycolmap, use_gpu: bool):
    if hasattr(pycolmap, "FeatureExtractionOptions"):
        options = pycolmap.FeatureExtractionOptions()
    elif hasattr(pycolmap, "SiftExtractionOptions"):
        options = pycolmap.SiftExtractionOptions()
    else:
        return None

    if hasattr(options, "use_gpu"):
        options.use_gpu = use_gpu
    return options


def _configure_matching_options(pycolmap, use_gpu: bool):
    if hasattr(pycolmap, "FeatureMatchingOptions"):
        options = pycolmap.FeatureMatchingOptions()
    elif hasattr(pycolmap, "SiftMatchingOptions"):
        options = pycolmap.SiftMatchingOptions()
    else:
        return None

    if hasattr(options, "use_gpu"):
        options.use_gpu = use_gpu
    return options


def run_pycolmap_pipeline(
    db_path: Path,
    images_dir: Path,
    sparse_dir: Path,
    use_gpu: bool,
) -> None:
    try:
        import pycolmap
    except ImportError as exc:
        raise RuntimeError("pycolmap が見つかりません。pip install pycolmap で導入してください。") from exc

    if hasattr(pycolmap, "Database"):
        try:
            db = pycolmap.Database.connect(str(db_path))
            db.close()
        except Exception:
            pass

    extraction_opts = _configure_extraction_options(pycolmap, use_gpu)
    matching_opts = _configure_matching_options(pycolmap, use_gpu)

    extract_kwargs = {
        "database_path": str(db_path),
        "image_path": str(images_dir),
    }
    if extraction_opts is not None:
        extract_kwargs["extraction_options"] = extraction_opts
    if hasattr(pycolmap, "CameraMode"):
        extract_kwargs["camera_mode"] = pycolmap.CameraMode.SINGLE

    if hasattr(pycolmap, "extract_features"):
        pycolmap.extract_features(**extract_kwargs)
    elif hasattr(pycolmap, "feature_extractor"):
        pycolmap.feature_extractor(**extract_kwargs)
    else:
        raise RuntimeError("pycolmap の特徴抽出APIが見つかりません。")

    match_kwargs = {"database_path": str(db_path)}
    if matching_opts is not None:
        match_kwargs["matching_options"] = matching_opts

    if hasattr(pycolmap, "match_exhaustive"):
        pycolmap.match_exhaustive(**match_kwargs)
    elif hasattr(pycolmap, "exhaustive_matcher"):
        pycolmap.exhaustive_matcher(**match_kwargs)
    else:
        raise RuntimeError("pycolmap の総当たりマッチングAPIが見つかりません。")

    map_kwargs = {
        "database_path": str(db_path),
        "image_path": str(images_dir),
        "output_path": str(sparse_dir),
    }

    if hasattr(pycolmap, "incremental_mapping"):
        pycolmap.incremental_mapping(**map_kwargs)
    elif hasattr(pycolmap, "mapper"):
        pycolmap.mapper(**map_kwargs)
    else:
        raise RuntimeError("pycolmap のマッピングAPIが見つかりません。")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "多視点動画から FreeTimeGS 用の COLMAP データを作るスクリプト\n"
            "例: python scripts/prepare_dataset.py ./dance ./dataset 0 60 60 2"
        )
    )
    parser.add_argument("videos_dir", type=Path, help="cam01.mp4 などが入ったディレクトリ")
    parser.add_argument("data_dir", type=Path, help="出力先データセットディレクトリ")
    parser.add_argument("frame_start", type=int, help="抽出開始フレーム (0-based)")
    parser.add_argument("frame_end", type=int, help="抽出終了フレーム (0-based, inclusive)")
    parser.add_argument("fps", type=int, nargs="?", default=30, help="抽出FPS")
    parser.add_argument("downsample", type=int, nargs="?", default=1, help="縮小倍率 (2なら1/2)")
    parser.add_argument("skip_extract", type=parse_bool, nargs="?", default=False, help="trueで抽出をスキップ")
    parser.add_argument("--use-gpu", type=parse_bool, default=True, help="pycolmapでGPUを使うか（デフォルト: true）")

    args = parser.parse_args()

    gpu_index = os.getenv("GPU_INDEX", "0")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_index)

    images_dir = args.data_dir / "images"
    sparse_dir = args.data_dir / "sparse"
    db_path = args.data_dir / "database.db"

    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_extract:
        print("[INFO] フレーム抽出をスキップします (skip_extract=true)")
    else:
        video_files = sorted(args.videos_dir.glob("*.mp4"))
        if not video_files:
            raise FileNotFoundError(f"{args.videos_dir} に mp4 が見つかりません")

        for video_path in video_files:
            cam_name = video_path.stem
            print(f"[INFO] 抽出開始: {video_path.name}")
            run_ffmpeg_extract(
                video_path=video_path,
                images_dir=images_dir,
                cam_name=cam_name,
                frame_start=args.frame_start,
                frame_end=args.frame_end,
                fps=args.fps,
                downsample=args.downsample,
            )

    if db_path.exists():
        print("[INFO] 既存の database.db を削除")
        db_path.unlink()

    run_pycolmap_pipeline(
        db_path=db_path,
        images_dir=images_dir,
        sparse_dir=sparse_dir,
        use_gpu=args.use_gpu,
    )

    cameras_bin = sparse_dir / "0" / "cameras.bin"
    images_bin = sparse_dir / "0" / "images.bin"
    points_bin = sparse_dir / "0" / "points3D.bin"

    if cameras_bin.exists() and images_bin.exists() and points_bin.exists():
        print(f"[SUCCESS] COLMAP データ準備完了: {sparse_dir / '0'}")
        print("          FreeTime_dataset.py を実行できる状態です。")
    else:
        raise RuntimeError("COLMAP の出力が見つかりません。ログを確認してください。")


if __name__ == "__main__":
    main()
