"""MegaSaM camera pose estimation helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "OpenCV (cv2) is required for MegaSaM preprocessing. Install it in the WM-retargeting env."
    ) from exc


@dataclass
class CameraFrame:
    """Per-frame camera calibration."""

    index: int
    intrinsic: np.ndarray  # 3x3
    cam_c2w: np.ndarray  # 4x4 homogeneous matrix
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        data["intrinsic"] = self.intrinsic.tolist()
        data["cam_c2w"] = self.cam_c2w.tolist()
        return data


@dataclass
class MegaSamCameraSolution:
    """Container for MegaSaM camera results."""

    scene: str
    frames: List[CameraFrame]
    source_npz: Path
    depths: Optional[np.ndarray] = None

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.frames)

    def to_json(self, output_path: Path) -> None:
        payload = {
            "scene": self.scene,
            "source_npz": str(self.source_npz),
            "frames": [frame.to_dict() for frame in self.frames],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))


def _call(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Optional[dict] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Thin wrapper that logs and executes subprocess calls."""

    print(f"[MegaSaM] Running: {' '.join(args)} (cwd={cwd})")
    proc = subprocess.run(args, cwd=str(cwd), env=env, check=check)
    return proc


def extract_frames(video_path: Path, output_dir: Path, *, fps: Optional[float] = None) -> List[Path]:
    """Extract frames from a video into ``output_dir`` using OpenCV."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)
    frame_paths: List[Path] = []
    index = 0
    success, frame = cap.read()
    while success:
        frame_path = output_dir / f"{index:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        frame_paths.append(frame_path)
        index += 1
        success, frame = cap.read()

    cap.release()
    print(f"Extracted {len(frame_paths)} frames to {output_dir}")
    meta_path = output_dir / "frames_meta.json"
    meta_path.write_text(json.dumps({"video": str(video_path), "fps": fps, "num_frames": len(frame_paths)}, indent=2))
    return frame_paths


def run_megasam_tracking(
    *,
    megasam_root: Path,
    frames_dir: Path,
    scene: str,
    working_dir: Path,
    enable_depth: bool = True,
    opt_focal: bool = True,
    gpu: str = "0",
) -> Path:
    """Run MegaSaM camera tracker and return output npz path."""

    megasam_root = Path(megasam_root).resolve()
    frames_dir = Path(frames_dir).resolve()
    working_dir = Path(working_dir).resolve()

    env = {"CUDA_VISIBLE_DEVICES": gpu, **dict(os.environ)}
    mono_depth_dir = Path(megasam_root) / "Depth-Anything" / "video_visualization" / scene
    unidepth_dir = Path(megasam_root) / "UniDepth" / "outputs"
    if enable_depth:
        mono_depth_dir.parent.mkdir(parents=True, exist_ok=True)
        depth_cmd = [
            sys.executable,
            "Depth-Anything/run_videos.py",
            "--encoder",
            "vitl",
            "--load-from",
            "Depth-Anything/checkpoints/depth_anything_vitl14.pth",
            "--img-path",
            str(frames_dir),
            "--outdir",
            str(mono_depth_dir),
        ]
        _call(depth_cmd, cwd=megasam_root, env=env)

        env["PYTHONPATH"] = str(Path(megasam_root) / "UniDepth") + ":" + env.get("PYTHONPATH", "")
        unidepth_cmd = [
            sys.executable,
            "UniDepth/scripts/demo_mega-sam.py",
            "--scene-name",
            scene,
            "--img-path",
            str(frames_dir),
            "--outdir",
            str(unidepth_dir),
        ]
        _call(unidepth_cmd, cwd=megasam_root, env=env)

    weights_path = megasam_root / "checkpoints" / "megasam_final.pth"

    cmd = [
        sys.executable,
        "camera_tracking_scripts/test_demo.py",
        "--datapath",
        str(frames_dir),
        "--disable_vis",
        "--weights",
        str(weights_path),
        "--scene_name",
        scene,
        "--mono_depth_path",
        str(mono_depth_dir.parent),
        "--metric_depth_path",
        str(unidepth_dir),
    ]

    if opt_focal:
        help_proc = subprocess.run(
            [sys.executable, "camera_tracking_scripts/test_demo.py", "--help"],
            cwd=str(megasam_root),
            env=env,
            capture_output=True,
            text=True,
        )
        if "--opt_focal" in help_proc.stdout:
            cmd.append("--opt_focal")
        else:
            print("[MegaSaM] --opt_focal unsupported; running with default intrinsics")
    _call(cmd, cwd=megasam_root, env=env)

    output_npz = Path(megasam_root) / "outputs" / f"{scene}_droid.npz"
    if not output_npz.exists():
        raise FileNotFoundError(f"MegaSaM output not found: {output_npz}")
    target_npz = working_dir / output_npz.name
    target_npz.parent.mkdir(parents=True, exist_ok=True)
    target_npz.write_bytes(output_npz.read_bytes())
    print(f"Copied MegaSaM solution to {target_npz}")
    return target_npz


def load_megasam_solution(npz_path: Path, *, timestamps: Optional[Iterable[float]] = None) -> MegaSamCameraSolution:
    """Load ``*_droid.npz`` into a structured ``MegaSamCameraSolution``."""

    data = np.load(npz_path)
    intr_raw = data["intrinsic"]
    poses = data["cam_c2w"]
    count = poses.shape[0]
    depths = data["depths"] if "depths" in data.files else None

    def _intrinsic_to_vector(mat: np.ndarray) -> np.ndarray:
        if mat.shape[-2:] != (3, 3):
            raise ValueError("Unsupported intrinsic matrix shape: {}".format(mat.shape))
        return np.stack([mat[..., 0, 0], mat[..., 1, 1], mat[..., 0, 2], mat[..., 1, 2]], axis=-1)

    if intr_raw.ndim == 2 and intr_raw.shape == (3, 3):
        intr = np.tile(_intrinsic_to_vector(intr_raw)[None, :], (count, 1))
    elif intr_raw.ndim == 1 and intr_raw.shape[0] == 4:
        intr = np.tile(intr_raw[None, :], (count, 1))
    elif intr_raw.ndim >= 2 and intr_raw.shape[0] == count:
        if intr_raw.shape[-1] == 4:
            intr = intr_raw
        elif intr_raw.shape[1:] == (3, 3):
            intr = _intrinsic_to_vector(intr_raw)
        else:
            raise ValueError("Unexpected intrinsic shape: {}".format(intr_raw.shape))
    else:
        raise ValueError("Intrinsic and pose arrays have inconsistent length")

    times: List[Optional[float]]
    if timestamps is None:
        times = [None] * count
    else:
        times_raw = list(timestamps)
        if len(times_raw) != count:
            raise ValueError("Timestamp list length mismatch")
        times = times_raw

    frames = []
    for idx in range(count):
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = intr[idx][0]
        intrinsic[1, 1] = intr[idx][1]
        intrinsic[0, 2] = intr[idx][2]
        intrinsic[1, 2] = intr[idx][3]
        intrinsic[2, 2] = 1.0
        cam_c2w = poses[idx]
        frames.append(CameraFrame(index=idx, intrinsic=intrinsic, cam_c2w=cam_c2w, timestamp=times[idx]))

    return MegaSamCameraSolution(
        scene=npz_path.stem.replace("_droid", ""),
        frames=frames,
        source_npz=npz_path,
        depths=depths,
    )


def estimate_cameras_from_video(
    video_path: Path,
    *,
    megasam_root: Path,
    scene: Optional[str] = None,
    working_dir: Optional[Path] = None,
    fps: Optional[float] = None,
    reuse_existing: bool = True,
) -> MegaSamCameraSolution:
    """Top-level helper that extracts frames and runs MegaSaM."""

    video_path = video_path.resolve()
    if scene is None:
        scene = video_path.stem
    if working_dir is None:
        working_dir = Path("video2real_outputs") / scene
    working_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = working_dir / "frames"
    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        frames_dir.mkdir(parents=True, exist_ok=True)
        extract_frames(video_path, frames_dir, fps=fps)
    else:
        print(f"Reusing extracted frames in {frames_dir}")

    output_npz = working_dir / f"{scene}_droid.npz"
    if not output_npz.exists() or not reuse_existing:
        output_npz = run_megasam_tracking(
            megasam_root=megasam_root,
            frames_dir=frames_dir,
            scene=scene,
            working_dir=working_dir,
        )
    else:
        print(f"Reusing cached MegaSaM solution {output_npz}")

    fps_meta = frames_dir / "frames_meta.json"
    timestamps = None
    if fps_meta.exists():
        meta = json.loads(fps_meta.read_text())
        if meta.get("fps"):
            timestamps = [i / meta["fps"] for i in range(meta["num_frames"])]

    return load_megasam_solution(output_npz, timestamps=timestamps)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run MegaSaM camera estimation on a video")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--scene", type=str, default=None, help="Scene name override")
    parser.add_argument("--megasam-root", type=Path, default=Path("mega-sam"), help="Path to mega-sam repo")
    parser.add_argument("--working-dir", type=Path, default=None, help="Cache directory for intermediate results")
    parser.add_argument("--no-reuse", action="store_true", help="Force rerun even if cached outputs exist")
    args = parser.parse_args(argv)

    solution = estimate_cameras_from_video(
        args.video,
        megasam_root=args.megasam_root,
        scene=args.scene,
        working_dir=args.working_dir,
        reuse_existing=not args.no_reuse,
    )
    print(f"Loaded MegaSaM solution with {len(solution)} frames from {solution.source_npz}")


if __name__ == "__main__":  # pragma: no cover
    main()
