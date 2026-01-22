"""Coordinate conversions for wrist pose."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
from pytransform3d import rotations

try:  # pragma: no cover - support script execution without package context
    from .cam_alignment import AlignedSequence
except ImportError:  # pragma: no cover
    from cam_alignment import AlignedSequence


R_G1_FROM_CAM = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def cam_to_g1_position(position_cam: np.ndarray) -> np.ndarray:
    return R_G1_FROM_CAM @ position_cam


def cam_to_g1_rotation(rotation_cam: np.ndarray) -> np.ndarray:
    return R_G1_FROM_CAM @ rotation_cam


def cam_quat_to_g1(quaternion_cam: np.ndarray) -> np.ndarray:
    R_cam = rotations.matrix_from_quaternion(quaternion_cam)
    R_g1 = cam_to_g1_rotation(R_cam)
    return rotations.quaternion_from_matrix(R_g1).astype(np.float32)


@dataclass
class WristPoseG1:
    frame_idx: int
    timestamp: float
    hand_type: str
    wrist_position_g1: np.ndarray
    wrist_quaternion_g1: np.ndarray
    rotation_matrix_g1: np.ndarray

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "hand_type": self.hand_type,
            "wrist_position_g1": self.wrist_position_g1.tolist(),
            "wrist_quaternion_g1": self.wrist_quaternion_g1.tolist(),
        }

    def to_payload(
        self,
        *,
        quat_format: str = "wxyz",
        include_matrix: bool = False,
    ) -> dict:
        quat = self.wrist_quaternion_g1.astype(np.float32)
        if quat_format == "xyzw":
            quat_out = np.concatenate((quat[1:], quat[:1]))
        elif quat_format == "wxyz":
            quat_out = quat
        else:  # pragma: no cover
            raise ValueError(f"Unsupported quaternion format: {quat_format}")

        payload = {
            "frame": self.frame_idx,
            "timestamp": self.timestamp,
            "hand_type": self.hand_type,
            "xyz": self.wrist_position_g1.astype(np.float32).tolist(),
            "quat": quat_out.tolist(),
        }

        if include_matrix:
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = self.rotation_matrix_g1.astype(np.float32)
            matrix[:3, 3] = self.wrist_position_g1.astype(np.float32)
            payload["matrix"] = matrix.tolist()

        return payload


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        return _lerp(q0, q1, t)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    if sin_theta_0 < 1e-6:
        return q0
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    s0 = np.sin(theta_0 - theta) / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return q0 * s0 + q1 * s1


def convert_sequence_wrist_poses(
    aligned: AlignedSequence,
    *,
    hand: Optional[str] = None,
    target_origin: Optional[Sequence[float]] = None,
    target_frame_indices: Optional[List[int]] = None,
    target_timestamps: Optional[List[float]] = None,
    default_origin: Optional[np.ndarray] = None,
) -> List[WristPoseG1]:
    hand_norm: Optional[str] = hand.lower() if hand is not None else None
    filtered_frames: Iterable = (
        frame
        for frame in aligned.frames
        if hand_norm is None or frame.hand_type.lower() == hand_norm
    )

    trace: List[WristPoseG1] = []
    for frame in filtered_frames:
        pos_cam = np.asarray(frame.wrist_cam0, dtype=np.float32)
        rot_cam = rotations.matrix_from_quaternion(
            np.asarray(frame.wrist_quat_cam0, dtype=np.float32)
        )
        pos_g1 = cam_to_g1_position(pos_cam)
        rot_g1 = cam_to_g1_rotation(rot_cam)
        quat_g1 = rotations.quaternion_from_matrix(rot_g1).astype(np.float32)
        trace.append(
            WristPoseG1(
                frame_idx=frame.frame_idx,
                timestamp=frame.timestamp,
                hand_type=frame.hand_type,
                wrist_position_g1=pos_g1.astype(np.float32),
                wrist_quaternion_g1=quat_g1,
                rotation_matrix_g1=rot_g1.astype(np.float32),
            )
        )

    # Store sorted trace for interpolation
    trace.sort(key=lambda pose: pose.frame_idx)
    frame_map = {pose.frame_idx: pose for pose in trace}

    if not trace:
        if target_frame_indices is None:
            return []
        if target_timestamps is None or len(target_timestamps) != len(target_frame_indices):
            target_timestamps = [float(i) for i in range(len(target_frame_indices))]
        default_pos = (
            default_origin.astype(np.float32)
            if default_origin is not None
            else np.zeros(3, dtype=np.float32)
        )
        default_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        default_rot = rotations.matrix_from_quaternion(default_quat)
        return [
            WristPoseG1(
                frame_idx=idx_target,
                timestamp=ts,
                hand_type=hand or "right",
                wrist_position_g1=default_pos.copy(),
                wrist_quaternion_g1=default_quat.copy(),
                rotation_matrix_g1=default_rot.copy(),
            )
            for idx_target, ts in zip(target_frame_indices, target_timestamps)
        ]

    if target_frame_indices is None:
        results = trace
    else:
        sorted_idx = [pose.frame_idx for pose in trace]
        sorted_pos = [pose.wrist_position_g1 for pose in trace]
        sorted_quat = [pose.wrist_quaternion_g1 for pose in trace]

        if target_timestamps is None or len(target_timestamps) != len(target_frame_indices):
            target_timestamps = [trace[0].timestamp for _ in target_frame_indices]

        first_target_idx = target_frame_indices[0]

        def sample_pose(idx_target: int, ts: float) -> WristPoseG1:
            if idx_target <= sorted_idx[0]:
                base = trace[0]
                pos_start = (
                    default_origin.astype(np.float32)
                    if default_origin is not None
                    else sorted_pos[0]
                )
                quat_start = sorted_quat[0]
                pos_end = sorted_pos[0]
                quat_end = sorted_quat[0]
                denom = max(sorted_idx[0] - first_target_idx, 1)
                t = (idx_target - first_target_idx) / denom
                t = np.clip(t, 0.0, 1.0)
                pos = _lerp(pos_start, pos_end, t)
                quat = _slerp(quat_start, quat_end, t)
                quat = quat / max(np.linalg.norm(quat), 1e-6)
            elif idx_target >= sorted_idx[-1]:
                base = trace[-1]
                pos = sorted_pos[-1]
                quat = sorted_quat[-1]
            else:
                lo = next((i for i, idx in enumerate(sorted_idx) if idx <= idx_target), 0)
                hi = next(i for i, idx in enumerate(sorted_idx) if idx >= idx_target)
                if lo == hi:
                    lo = max(0, hi - 1)
                idx_lo = sorted_idx[lo]
                idx_hi = sorted_idx[hi]
                pose_lo = trace[lo]
                pose_hi = trace[hi]
                if idx_hi == idx_lo:
                    pos = pose_lo.wrist_position_g1
                    quat = pose_lo.wrist_quaternion_g1
                    base = pose_lo
                else:
                    t = (idx_target - idx_lo) / (idx_hi - idx_lo)
                    pos = _lerp(pose_lo.wrist_position_g1, pose_hi.wrist_position_g1, t)
                    quat = _slerp(pose_lo.wrist_quaternion_g1, pose_hi.wrist_quaternion_g1, t)
                    quat = quat / max(np.linalg.norm(quat), 1e-6)
                    base = pose_lo
            rot = rotations.matrix_from_quaternion(quat)
            return WristPoseG1(
                frame_idx=idx_target,
                timestamp=ts,
                hand_type=base.hand_type,
                wrist_position_g1=pos.astype(np.float32),
                wrist_quaternion_g1=quat.astype(np.float32),
                rotation_matrix_g1=rot.astype(np.float32),
            )

        results = [
            sample_pose(idx_target, ts)
            for idx_target, ts in zip(target_frame_indices, target_timestamps)
        ]

    if target_origin is not None and results:
        target = np.asarray(target_origin, dtype=np.float32)
        offset = target - results[0].wrist_position_g1
        for pose in results:
            pose.wrist_position_g1 = pose.wrist_position_g1 + offset

    if default_origin is not None and results:
        set_default = False
        if target_frame_indices is None:
            set_default = False
        else:
            first_target = target_frame_indices[0]
            has_exact = any(p.frame_idx == first_target for p in trace)
            set_default = not has_exact
        if set_default:
            results[0].wrist_position_g1 = default_origin.astype(np.float32)

    return results


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Convert wrist poses to G1 coordinate frame")
    parser.add_argument("aligned_json", type=Path, help="Aligned hand JSON")
    parser.add_argument("--output", type=Path, default=Path("video2real_outputs/wrist_g1.json"))
    parser.add_argument(
        "--hand",
        type=str,
        choices=("left", "right"),
        default=None,
        help="Filter to a single hand before conversion",
    )
    parser.add_argument(
        "--target-origin",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Translate the first frame to this G1 position (applied to all frames)",
    )

    aligned = AlignedSequence.from_json(args.aligned_json)
    poses = convert_sequence_wrist_poses(
        aligned,
        hand=args.hand,
        target_origin=args.target_origin,
    )
    payload = [
        pose.to_payload(
            quat_format=args.quat_format,
            include_matrix=args.include_matrix,
        )
        for pose in poses
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"Converted {len(poses)} wrist poses to G1 frame: {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
