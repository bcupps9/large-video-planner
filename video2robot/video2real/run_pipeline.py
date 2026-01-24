"""Full video-to-Inspire pipeline orchestrator."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import cv2

from dex_retargeting.constants import (
    HandType,
    RetargetingType,
    RobotName,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig

from .camera_estimation import estimate_cameras_from_video
from .hamer_extractor import HAMERExtractor
from .cam_alignment import align_hamer_to_cam0
from .wrist_frame import convert_sequence_wrist_poses
from .convert import convert_qpos_to_dof


def _relocate_megasam_outputs(output_dir: Path, cameras) -> None:
    """Move MegaSaM artifacts (frames, npz) next to pipeline outputs."""

    megasam_dir = output_dir / "megasam"
    if not megasam_dir.exists():
        return

    frames_dir = megasam_dir / "frames"
    if frames_dir.exists():
        dest_frames = output_dir / "frames"
        if dest_frames.exists():
            shutil.rmtree(dest_frames)
        shutil.move(str(frames_dir), str(dest_frames))

    source_npz = Path(cameras.source_npz)
    if source_npz.exists():
        dest_npz = output_dir / source_npz.name
        if dest_npz.exists():
            dest_npz.unlink()
        shutil.move(str(source_npz), str(dest_npz))
        cameras.source_npz = dest_npz

    shutil.rmtree(megasam_dir, ignore_errors=True)


def run_pipeline(
    video_path: Path,
    *,
    megasam_root: Path,
    output_dir: Path,
    scene: Optional[str] = None,
    reuse_megasam: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[video2real] Estimating cameras with MegaSaM...")
    cameras = estimate_cameras_from_video(
        video_path,
        megasam_root=megasam_root,
        scene=scene,
        working_dir=output_dir / "megasam",
        reuse_existing=reuse_megasam,
    )
    _relocate_megasam_outputs(output_dir, cameras)

    print("[video2real] Running HAMER hand reconstruction...")
    hamer_output = output_dir / "hamer.json"
    extractor = HAMERExtractor()
    hamer_seq = extractor.run(video_path)
    hamer_seq.to_json(hamer_output)

    print("[video2real] Aligning to cam0/world frame...")
    aligned = align_hamer_to_cam0(hamer_seq, cameras)
    aligned_json = output_dir / "aligned.json"
    aligned.to_json(aligned_json)

    print("[video2real] Retargeting vectors to Inspire joints...")
    asset_dir = (
        Path(__file__).resolve().parent.parent
        / "dex-retargeting"
        / "assets"
        / "robots"
        / "hands"
    )
    RetargetingConfig.set_default_urdf_dir(str(asset_dir))
    retarget_json = output_dir / "retarget_vector.json"
    finger_indices, finger_timestamps, qpos_list = _retarget_fingers_with_detect(
        video_path, retarget_json
    )

    cmd_values = convert_qpos_to_dof(qpos_list)
    cmd_json = output_dir / "cmd.json"
    cmd_json.write_text(json.dumps({"cmd": cmd_values}, indent=2))

    print("[video2real] Converting wrist pose to G1 frame...")
    wrist_poses = convert_sequence_wrist_poses(
        aligned,
        hand="right",
        target_frame_indices=finger_indices,
        target_timestamps=finger_timestamps,
        default_origin=np.array([0.25, -0.4, 0.1], dtype=np.float32),
    )
    wrist_json = output_dir / "wrist_g1.json"
    wrist_payload = {"frames": [pose.to_dict() for pose in wrist_poses]}
    wrist_json.write_text(json.dumps(wrist_payload, indent=2))

    final_payload = {
        "video": str(video_path),
        "megasam_npz": str(cameras.source_npz),
        "hamer_json": str(hamer_output),
        "aligned_json": str(aligned_json),
        "retarget_json": str(retarget_json),
        "cmd_json": str(cmd_json),
        "wrist_g1_json": str(wrist_json),
    }
    (output_dir / "summary.json").write_text(json.dumps(final_payload, indent=2))
    print(f"[video2real] Pipeline complete. Summary: {output_dir / 'summary.json'}")


def _retarget_fingers_with_detect(
    video_path: Path, retarget_json: Path
) -> tuple[list[int], list[float], list[list[float]]]:
    detect_script = (
        Path(__file__).resolve().parent.parent
        / "dex-retargeting"
        / "example"
        / "vector_retargeting"
        / "detect_from_video.py"
    )
    if not detect_script.exists():
        raise FileNotFoundError(f"detect_from_video.py not found at {detect_script}")

    retarget_pkl = retarget_json.parent / f"{retarget_json.stem}_tmp.pkl"
    cmd = [
        sys.executable,
        str(detect_script),
        "--robot-name",
        "inspire",
        "--video-path",
        str(video_path),
        "--output-path",
        str(retarget_pkl),
        "--retargeting-type",
        "vector",
        "--hand-type",
        "right",
    ]
    subprocess.run(cmd, check=True, cwd=detect_script.parent)

    try:
        with retarget_pkl.open("rb") as f:
            payload = pickle.load(f)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to read retarget pickle {retarget_pkl}: {exc}")
    finally:
        if retarget_pkl.exists():
            retarget_pkl.unlink()

    qpos_list = payload.get("data", [])
    joint_names = payload.get("meta_data", {}).get("joint_names", [])

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    if cap.isOpened():
        cap.release()
    if fps <= 0:
        fps = 30.0

    retarget_frames = []
    frame_indices = []
    timestamps = []
    for idx, qpos in enumerate(qpos_list):
        if hasattr(qpos, "tolist"):
            qpos = qpos.tolist()
        retarget_frames.append(
            {
                "frame_idx": idx,
                "timestamp": idx / fps,
                "joint_angles": qpos,
            }
        )
        frame_indices.append(idx)
        timestamps.append(idx / fps)

    retarget_json.write_text(
        json.dumps({"frames": retarget_frames, "joint_names": joint_names}, indent=2)
    )
    return frame_indices, timestamps, qpos_list


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the full video→Inspire pipeline")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--megasam-root", type=Path, default=Path("mega-sam"))
    parser.add_argument("--output-dir", type=Path, default=Path("video2real_outputs"))
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--no-reuse", action="store_true", help="Force MegaSaM rerun")
    args = parser.parse_args(argv)

    run_pipeline(
        args.video,
        megasam_root=args.megasam_root,
        output_dir=args.output_dir,
        scene=args.scene,
        reuse_megasam=not args.no_reuse,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
