#!/usr/bin/env python3
"""
FreeTimeGS 対話型パイプラインCLI

動画 → フレーム抽出 → COLMAP → RoMa三角測量 → キーフレーム結合 → トレーニング
の全ステップを対話的にガイドする。

Usage:
    python freetime_cli.py
    python freetime_cli.py --video-dir ./dance --data-dir ./dataset/my_scene
    python freetime_cli.py --yes          # 全プロンプトに自動応答（CI向け）
    python freetime_cli.py --gpu-id 2     # GPU IDを指定
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

# ============================================================
# Globals
# ============================================================
console = Console()

# プロジェクトルート（この CLI スクリプトのあるディレクトリ）
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

# ステップ表示用の定数
STEP_STYLES = {
    "done": "[bold green]✅ 完了[/]",
    "skip": "[bold yellow]⏭️  スキップ[/]",
    "run": "[bold cyan]🔄 実行中[/]",
    "fail": "[bold red]❌ 失敗[/]",
}


# ============================================================
# ユーティリティ
# ============================================================
def banner():
    """起動バナーを表示する。"""
    text = Text.from_markup(
        "[bold magenta]FreeTimeGS[/] [dim]— 4D Gaussian Splatting Pipeline[/]\n"
        "[dim]動画から4Dガウシアンスプラッティングモデルを構築します[/]"
    )
    console.print(Panel(text, border_style="bright_magenta", padding=(1, 2)))
    console.print()


def step_header(num: int, title: str):
    """ステップヘッダーを表示する。"""
    console.rule(f"[bold bright_cyan]Step {num}[/]  {title}", style="bright_cyan")
    console.print()


def run_cmd(cmd: list[str], env: dict | None = None, cwd: str | None = None) -> int:
    """
    コマンドをサブプロセスで実行し、リアルタイムで stdout/stderr を転送する。
    戻り値: returncode
    """
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    console.print(f"[dim]$ {' '.join(str(c) for c in cmd)}[/]")
    console.print()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged_env,
        cwd=cwd or str(PROJECT_ROOT),
    )
    for line in proc.stdout:
        console.print(f"  [dim]{line.rstrip()}[/]")
    proc.wait()
    return proc.returncode


def confirm(message: str, auto_yes: bool, default: bool = True) -> bool:
    """--yes モードではデフォルト値を即返す。"""
    if auto_yes:
        return default
    return Confirm.ask(message, default=default)


def fail_and_exit(step: str, returncode: int):
    """ステップ失敗時のメッセージを表示して終了する。"""
    console.print(f"\n{STEP_STYLES['fail']}  {step} がエラーで終了しました (code={returncode})")
    console.print("[yellow]ログを確認して問題を修正してから再実行してください。[/]")
    sys.exit(returncode)


# ============================================================
# Step 0: 動画ディレクトリとパラメータの設定
# ============================================================
def step0_configure(args) -> dict:
    """対話でパラメータを確定し、設定辞書を返す。"""
    step_header(0, "プロジェクト設定")

    # --- 動画ディレクトリ ---
    if args.video_dir:
        video_dir = Path(args.video_dir).resolve()
    else:
        raw = Prompt.ask(
            "[bold]動画ディレクトリのパスを入力してください[/]",
            default=str(PROJECT_ROOT / "dance"),
        )
        video_dir = Path(raw).resolve()

    if not video_dir.exists():
        console.print(f"[red]エラー: ディレクトリが見つかりません: {video_dir}[/]")
        sys.exit(1)

    # mp4 ファイルを列挙
    mp4_files = sorted(video_dir.glob("*.mp4"))
    if not mp4_files:
        console.print(f"[red]エラー: {video_dir} に .mp4 ファイルがありません[/]")
        sys.exit(1)

    table = Table(title="検出された動画ファイル", border_style="bright_blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("ファイル名", style="cyan")
    table.add_column("サイズ", justify="right", style="green")
    for i, f in enumerate(mp4_files):
        size_mb = f.stat().st_size / (1024 * 1024)
        table.add_row(str(i), f.name, f"{size_mb:.1f} MB")
    console.print(table)
    console.print()

    num_cameras = len(mp4_files)
    console.print(f"[bold]カメラ台数:[/] [bright_green]{num_cameras}[/]")

    # --- データ出力ディレクトリ ---
    if args.data_dir:
        data_dir = Path(args.data_dir).resolve()
    else:
        default_data = str(PROJECT_ROOT / "dataset" / video_dir.stem)
        raw = Prompt.ask(
            "[bold]データ出力ディレクトリのパスを入力してください[/]",
            default=default_data,
        )
        data_dir = Path(raw).resolve()

    # --- 結果出力ディレクトリ ---
    if args.result_dir:
        result_dir = Path(args.result_dir).resolve()
    else:
        default_result = str(PROJECT_ROOT / "results" / video_dir.stem)
        raw = Prompt.ask(
            "[bold]トレーニング結果の出力先を入力してください[/]",
            default=default_result,
        )
        result_dir = Path(raw).resolve()

    # --- フレーム数 / FPS ---
    start_frame = args.start_frame or 0
    num_frames = args.num_frames or 60
    fps = args.fps or 60
    keyframe_step = args.keyframe_step or 5
    max_steps = args.max_steps or 50000

    # --- サマリー表示 ---
    console.print()
    summary = Table(title="パイプライン設定", border_style="bright_magenta", show_lines=True)
    summary.add_column("項目", style="bold")
    summary.add_column("値", style="bright_white")
    summary.add_row("動画ディレクトリ", str(video_dir))
    summary.add_row("カメラ台数", str(num_cameras))
    summary.add_row("データ出力先", str(data_dir))
    summary.add_row("結果出力先", str(result_dir))
    summary.add_row("開始フレーム", str(start_frame))
    summary.add_row("フレーム数", str(num_frames))
    summary.add_row("FPS", str(fps))
    summary.add_row("キーフレーム間隔", str(keyframe_step))
    summary.add_row("トレーニングステップ数", str(max_steps))
    summary.add_row("GPU ID", str(args.gpu_id))
    console.print(summary)
    console.print()

    if not confirm("この設定で進めますか？", args.yes):
        console.print("[yellow]中断しました。[/]")
        sys.exit(0)

    return {
        "video_dir": video_dir,
        "data_dir": data_dir,
        "result_dir": result_dir,
        "num_cameras": num_cameras,
        "start_frame": start_frame,
        "num_frames": num_frames,
        "fps": fps,
        "keyframe_step": keyframe_step,
        "max_steps": max_steps,
        "gpu_id": args.gpu_id,
        "auto_yes": args.yes,
    }


# ============================================================
# Step 1: フレーム抽出
# ============================================================
def step1_extract_frames(cfg: dict) -> str:
    """MP4 → PNG フレーム抽出。状態を返す ("done"/"skip")。"""
    step_header(1, "フレーム抽出（MP4 → PNG）")

    images_dir = cfg["data_dir"] / "images"
    num_cameras = cfg["num_cameras"]
    num_frames = cfg["num_frames"]

    # 既存チェック: 全カメラフォルダに期待枚数のPNGがあるか
    existing_count = 0
    if images_dir.exists():
        for i in range(num_cameras):
            cam_dir = images_dir / f"{i:04d}"
            if cam_dir.exists():
                pngs = list(cam_dir.glob("*.png"))
                if len(pngs) >= num_frames:
                    existing_count += 1

    if existing_count == num_cameras:
        console.print(
            f"[green]全 {num_cameras} カメラのフレームが既に存在します[/] "
            f"({images_dir})"
        )
        if confirm("既存フレームをそのまま使いますか？（Noで再抽出）", cfg["auto_yes"]):
            console.print(STEP_STYLES["skip"])
            return "skip"
        # 再抽出: images ディレクトリを削除
        console.print("[yellow]既存フレームを削除して再抽出します...[/]")
        shutil.rmtree(images_dir)
    elif existing_count > 0:
        console.print(
            f"[yellow]{existing_count}/{num_cameras} カメラのフレームが部分的に存在します[/]"
        )
        if confirm("既存分を活かして不足分のみ抽出しますか？（Noで全削除して再抽出）", cfg["auto_yes"]):
            pass  # 既存はそのままで extract_selfcap_frames.sh 側のスキップ機能に任せる
        else:
            console.print("[yellow]既存フレームを削除して再抽出します...[/]")
            shutil.rmtree(images_dir)

    console.print(STEP_STYLES["run"])
    rc = run_cmd([
        "bash", "scripts/extract_selfcap_frames.sh",
        "--video-dir", str(cfg["video_dir"]),
        "--output-dir", str(images_dir),
        "--num-cameras", str(num_cameras),
        "--num-frames", str(num_frames),
        "--start-frame", str(cfg["start_frame"]),
        "--fps", str(cfg["fps"]),
    ])
    if rc != 0:
        fail_and_exit("フレーム抽出", rc)

    console.print(STEP_STYLES["done"])
    return "done"


# ============================================================
# Step 2: COLMAP キャリブレーション
# ============================================================
def step2_colmap(cfg: dict) -> str:
    """COLMAP キャリブレーション。"""
    step_header(2, "COLMAP キャリブレーション")

    colmap_sparse = cfg["data_dir"] / "colmap" / "sparse" / "0"
    cameras_bin = colmap_sparse / "cameras.bin"
    cameras_txt = colmap_sparse / "cameras.txt"

    if cameras_bin.exists() or cameras_txt.exists():
        console.print(f"[green]COLMAPモデルが既に存在します[/] ({colmap_sparse})")

        # カメラファイル情報を表示
        for f in colmap_sparse.iterdir():
            size_kb = f.stat().st_size / 1024
            console.print(f"  [dim]{f.name}[/]  [bright_green]{size_kb:.1f} KB[/]")

        if confirm("既存のCOLMAPモデルをそのまま使いますか？（Noで再構築）", cfg["auto_yes"]):
            console.print(STEP_STYLES["skip"])
            return "skip"
        # 再構築: colmap ディレクトリごと削除
        console.print("[yellow]既存のCOLMAPモデルを削除して再構築します...[/]")
        shutil.rmtree(cfg["data_dir"] / "colmap")

    console.print(STEP_STYLES["run"])
    rc = run_cmd([
        "bash", "scripts/run_colmap_calib.sh",
        "--data-dir", str(cfg["data_dir"]),
        "--num-cameras", str(cfg["num_cameras"]),
        "--image-ext", "png",
    ])
    if rc != 0:
        fail_and_exit("COLMAP キャリブレーション", rc)

    console.print(STEP_STYLES["done"])
    return "done"


# ============================================================
# Step 3: RoMa 三角測量
# ============================================================
def step3_roma(cfg: dict) -> str:
    """RoMa 三角測量 → per-frame NPY。"""
    step_header(3, "RoMa 三角測量")

    triangulation_dir = cfg["data_dir"] / "triangulation"
    num_frames = cfg["num_frames"]

    # 既存NPYの数を確認
    existing_npy = 0
    if triangulation_dir.exists():
        existing_npy = len(list(triangulation_dir.glob("points3d_frame*.npy")))

    if existing_npy >= num_frames:
        console.print(
            f"[green]三角測量NPYが既に {existing_npy} フレーム分存在します[/] "
            f"({triangulation_dir})"
        )
        if confirm("既存のNPYをそのまま使いますか？（Noで全削除して再実行）", cfg["auto_yes"]):
            console.print(STEP_STYLES["skip"])
            return "skip"
        console.print("[yellow]既存の三角測量結果を削除して再実行します...[/]")
        shutil.rmtree(triangulation_dir)
    elif existing_npy > 0:
        console.print(
            f"[yellow]三角測量NPYが {existing_npy}/{num_frames} フレーム分のみ存在します[/]"
        )
        if not confirm("全削除して最初から再実行しますか？（Noで中断）", cfg["auto_yes"]):
            console.print("[yellow]中断しました。[/]")
            sys.exit(0)
        shutil.rmtree(triangulation_dir)

    images_dir = cfg["data_dir"] / "images"
    colmap_model = cfg["data_dir"] / "colmap" / "sparse" / "0"

    # 環境変数の設定（MPS/macOS対策）
    extra_env = {}
    extra_env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    extra_env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    console.print(STEP_STYLES["run"])
    rc = run_cmd(
        [
            str(VENV_PYTHON), "scripts/roma_triangulate_to_npy.py",
            "--images-dir", str(images_dir),
            "--colmap-model", str(colmap_model),
            "--output-dir", str(triangulation_dir),
            "--frame-start", "0",
            "--frame-end", str(num_frames - 1),
            "--frame-step", "1",
            "--ref-cam", "0000",
            "--device", "auto",
            "--certainty", "0.3",
            "--min-depth", "1e-4",
            "--image-scale", "1.0",
            "--voxel-size", "0",
        ],
        env=extra_env,
    )
    if rc != 0:
        fail_and_exit("RoMa 三角測量", rc)

    console.print(STEP_STYLES["done"])
    return "done"


# ============================================================
# Step 4: キーフレーム結合
# ============================================================
def step4_combine(cfg: dict) -> tuple[str, Path]:
    """NPY → NPZ キーフレーム結合。(状態, npzパス) を返す。"""
    step_header(4, "キーフレーム結合（NPY → NPZ）")

    num_frames = cfg["num_frames"]
    keyframe_step = cfg["keyframe_step"]
    triangulation_dir = cfg["data_dir"] / "triangulation"
    start_frame = cfg["start_frame"]
    npz_path = cfg["data_dir"] / f"keyframes_{num_frames}frames_start{start_frame}_step{keyframe_step}.npz"

    if npz_path.exists():
        size_mb = npz_path.stat().st_size / (1024 * 1024)
        console.print(f"[green]NPZファイルが既に存在します[/]: {npz_path.name} ({size_mb:.1f} MB)")

        # NPZのメタ情報を表示
        try:
            import numpy as np
            with np.load(str(npz_path), allow_pickle=True) as data:
                table = Table(title="NPZ メタデータ", border_style="bright_blue")
                table.add_column("キー", style="cyan")
                table.add_column("shape", style="bright_white")
                table.add_column("dtype", style="dim")
                for key in sorted(data.files):
                    arr = data[key]
                    if hasattr(arr, "shape"):
                        table.add_row(key, str(arr.shape), str(arr.dtype))
                    else:
                        table.add_row(key, str(arr), "scalar")
                console.print(table)
        except Exception:
            pass

        if confirm("既存のNPZをそのまま使いますか？（Noで再生成）", cfg["auto_yes"]):
            console.print(STEP_STYLES["skip"])
            return "skip", npz_path

        # 再生成: NPZ と NPY 両方を削除
        console.print("[yellow]既存のNPZとNPYを削除して再生成します...[/]")
        npz_path.unlink()
        if triangulation_dir.exists():
            shutil.rmtree(triangulation_dir)
        # NPYを再生成するため RoMa ステップを再実行
        console.print("[yellow]NPYが削除されたため、RoMa 三角測量を再実行します...[/]")
        step3_roma(cfg)

    console.print(STEP_STYLES["run"])
    rc = run_cmd([
        str(VENV_PYTHON), "src/combine_frames_fast_keyframes.py",
        "--input-dir", str(triangulation_dir),
        "--output-path", str(npz_path),
        "--frame-start", "0",
        "--frame-end", str(num_frames - 1),
        "--keyframe-step", str(keyframe_step),
    ])
    if rc != 0:
        fail_and_exit("キーフレーム結合", rc)

    console.print(STEP_STYLES["done"])
    return "done", npz_path


# ============================================================
# Step 5: トレーニング
# ============================================================
def step5_train(cfg: dict, npz_path: Path) -> str:
    """4D Gaussian Splatting のトレーニングを実行する。"""
    step_header(5, "トレーニング")

    num_frames = cfg["num_frames"]
    max_steps = cfg["max_steps"]
    result_dir = cfg["result_dir"]
    data_dir = cfg["data_dir"]
    gpu_id = cfg["gpu_id"]

    # 既存チェックポイントの確認
    ckpt_dir = result_dir / "ckpts"
    if ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"))
        if ckpts:
            latest = ckpts[-1]
            size_mb = latest.stat().st_size / (1024 * 1024)
            console.print(
                f"[green]既存チェックポイントが見つかりました:[/] {latest.name} ({size_mb:.1f} MB)"
            )
            if confirm("既存の結果をスキップしますか？（Noで最初からトレーニング）", cfg["auto_yes"]):
                console.print(STEP_STYLES["skip"])
                return "skip"
            # 結果ディレクトリを消してクリーンスタート
            console.print("[yellow]既存の結果を削除してクリーンスタートします...[/]")
            shutil.rmtree(result_dir)

    # サマリー表示
    console.print()
    table = Table(title="トレーニング設定", border_style="bright_green", show_lines=True)
    table.add_column("項目", style="bold")
    table.add_column("値", style="bright_white")
    table.add_row("Config", "default_keyframe")
    table.add_row("data-dir", str(data_dir))
    table.add_row("init-npz-path", str(npz_path))
    table.add_row("result-dir", str(result_dir))
    table.add_row("フレーム範囲", f"0 → {num_frames + 1}")
    table.add_row("max-steps", str(max_steps))
    table.add_row("GPU (CUDA_VISIBLE_DEVICES)", str(gpu_id))
    console.print(table)
    console.print()

    if not confirm("トレーニングを開始しますか？", cfg["auto_yes"]):
        console.print("[yellow]トレーニングをスキップしました。[/]")
        return "skip"

    console.print(STEP_STYLES["run"])
    rc = run_cmd(
        [
            str(VENV_PYTHON), "src/simple_trainer_freetime_4d_pure_relocation.py",
            "default_keyframe",
            "--data-dir", str(data_dir),
            "--init-npz-path", str(npz_path),
            "--result-dir", str(result_dir),
            "--start-frame", "0",
            "--end-frame", str(num_frames + 1),
            "--max-steps", str(max_steps),
            "--eval-steps", str(max_steps),
            "--save-steps", str(max_steps),
        ],
        env={"CUDA_VISIBLE_DEVICES": str(gpu_id)},
    )
    if rc != 0:
        fail_and_exit("トレーニング", rc)

    console.print(STEP_STYLES["done"])
    return "done"


# ============================================================
# 完了サマリー
# ============================================================
def print_summary(cfg: dict, npz_path: Path, results: dict[str, str]):
    """パイプライン完了サマリーを表示する。"""
    console.print()
    console.rule("[bold bright_green]パイプライン完了", style="bright_green")
    console.print()

    table = Table(border_style="bright_green", show_lines=True)
    table.add_column("ステップ", style="bold")
    table.add_column("結果")
    step_names = [
        "1. フレーム抽出",
        "2. COLMAP",
        "3. RoMa 三角測量",
        "4. キーフレーム結合",
        "5. トレーニング",
    ]
    for name, key in zip(step_names, ["extract", "colmap", "roma", "combine", "train"]):
        status = results.get(key, "—")
        if status == "done":
            table.add_row(name, STEP_STYLES["done"])
        elif status == "skip":
            table.add_row(name, STEP_STYLES["skip"])
        else:
            table.add_row(name, f"[dim]{status}[/]")
    console.print(table)

    console.print()
    paths = Table(title="出力パス", border_style="bright_blue")
    paths.add_column("種別", style="bold")
    paths.add_column("パス", style="cyan")
    paths.add_row("画像", str(cfg["data_dir"] / "images"))
    paths.add_row("COLMAP", str(cfg["data_dir"] / "colmap" / "sparse" / "0"))
    paths.add_row("三角測量", str(cfg["data_dir"] / "triangulation"))
    paths.add_row("NPZ", str(npz_path))
    paths.add_row("トレーニング結果", str(cfg["result_dir"]))
    console.print(paths)
    console.print()


# ============================================================
# メイン
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="FreeTimeGS 対話型パイプラインCLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例:\n"
            "  python freetime_cli.py\n"
            "  python freetime_cli.py --video-dir ./dance --data-dir ./dataset/dance\n"
            "  python freetime_cli.py --yes --gpu-id 0\n"
        ),
    )
    parser.add_argument("--video-dir", type=str, default=None,
                        help="入力動画ディレクトリ（省略時は対話で入力）")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="データ出力ディレクトリ（省略時は対話で入力）")
    parser.add_argument("--result-dir", type=str, default=None,
                        help="トレーニング結果の出力先（省略時は対話で入力）")
    parser.add_argument("--gpu-id", type=int, default=0,
                        help="使用するGPU ID (default: 0)")
    parser.add_argument("--start-frame", type=int, default=None,
                        help="動画の抽出開始フレーム番号 (default: 0)。例: --start-frame 60 で61番目のフレームから抽出")
    parser.add_argument("--num-frames", type=int, default=None,
                        help="抽出フレーム数 (default: 60)")
    parser.add_argument("--fps", type=int, default=None,
                        help="抽出FPS (default: 60)")
    parser.add_argument("--keyframe-step", type=int, default=None,
                        help="キーフレーム間隔 (default: 5)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="トレーニングステップ数 (default: 50000)")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="全ての確認プロンプトに自動でYesを返す")
    return parser.parse_args()


def main():
    args = parse_args()

    banner()

    # Step 0: 設定
    cfg = step0_configure(args)

    results = {}

    # Step 1: フレーム抽出
    results["extract"] = step1_extract_frames(cfg)
    console.print()

    # Step 2: COLMAP
    results["colmap"] = step2_colmap(cfg)
    console.print()

    # Step 3: RoMa 三角測量
    results["roma"] = step3_roma(cfg)
    console.print()

    # Step 4: キーフレーム結合
    status, npz_path = step4_combine(cfg)
    results["combine"] = status
    console.print()

    # Step 5: トレーニング
    results["train"] = step5_train(cfg, npz_path)

    # サマリー
    print_summary(cfg, npz_path, results)


if __name__ == "__main__":
    main()
