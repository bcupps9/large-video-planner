"""Align HAMER outputs to the MegaSaM cam0/world frame."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from pytransform3d import rotations

try:  # pragma: no cover - allow running as module or script
    from .camera_estimation import CameraFrame, MegaSamCameraSolution, load_megasam_solution
    from .hamer_extractor import HAMERSequenceResult
except ImportError:  # pragma: no cover
    from camera_estimation import CameraFrame, MegaSamCameraSolution, load_megasam_solution
    from hamer_extractor import HAMERSequenceResult


@dataclass
class AlignedHandFrame:
    frame_idx: int
    timestamp: float
    hand_type: str
    confidence: float
    wrist_cam: np.ndarray
    wrist_cam0: np.ndarray
    wrist_quat_cam: np.ndarray
    wrist_quat_cam0: np.ndarray
    keypoints_cam: np.ndarray
    keypoints_cam0: np.ndarray

    def to_dict(self) -> dict:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "hand_type": self.hand_type,
            "confidence": self.confidence,
            "wrist_cam": self.wrist_cam.tolist(),
            "wrist_cam0": self.wrist_cam0.tolist(),
            "wrist_quat_cam": self.wrist_quat_cam.tolist(),
            "wrist_quat_cam0": self.wrist_quat_cam0.tolist(),
            "keypoints_cam": self.keypoints_cam.tolist(),
            "keypoints_cam0": self.keypoints_cam0.tolist(),
        }


@dataclass
class AlignedSequence:
    frames: List[AlignedHandFrame]

    def filter_by_hand(self, hand: str) -> List[AlignedHandFrame]:
        return [frame for frame in self.frames if frame.hand_type.lower() == hand.lower()]

    def to_json(self, output_path: Path) -> None:
        payload = {"frames": [frame.to_dict() for frame in self.frames]}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, json_path: Path) -> "AlignedSequence":
        data = json.loads(json_path.read_text())
        frames = []
        for item in data["frames"]:
            frames.append(
                AlignedHandFrame(
                    frame_idx=item["frame_idx"],
                    timestamp=item["timestamp"],
                    hand_type=item["hand_type"],
                    confidence=item["confidence"],
                    wrist_cam=np.asarray(item["wrist_cam"], dtype=np.float32),
                    wrist_cam0=np.asarray(item["wrist_cam0"], dtype=np.float32),
                    wrist_quat_cam=np.asarray(item["wrist_quat_cam"], dtype=np.float32),
                    wrist_quat_cam0=np.asarray(item["wrist_quat_cam0"], dtype=np.float32),
                    keypoints_cam=np.asarray(item["keypoints_cam"], dtype=np.float32),
                    keypoints_cam0=np.asarray(item["keypoints_cam0"], dtype=np.float32),
                )
            )
        return cls(frames=frames)


def align_hamer_to_cam0(
    hamer_seq: HAMERSequenceResult,
    cameras: MegaSamCameraSolution,
) -> AlignedSequence:
    """Convert HAMER results into the cam0/world coordinate frame using MegaSaM depths."""

    camera_lookup: Dict[int, CameraFrame] = {frame.index: frame for frame in cameras.frames}
    depths = cameras.depths
    aligned_frames: List[AlignedHandFrame] = []

    if hamer_seq.frame_size is not None:
        orig_width, orig_height = map(float, hamer_seq.frame_size)
    else:
        orig_width = orig_height = None

    depth_scale_u: Optional[float] = None
    depth_scale_v: Optional[float] = None
    if depths is not None and orig_width and orig_height:
        depth_height, depth_width = depths.shape[1:]
        depth_scale_u = depth_width / orig_width
        depth_scale_v = depth_height / orig_height

    for hand_frame in hamer_seq.frames:
        cam_frame = camera_lookup.get(hand_frame.frame_idx)
        if cam_frame is None:
            continue

        depth_map = None
        if depths is not None and 0 <= hand_frame.frame_idx < depths.shape[0]:
            depth_map = depths[hand_frame.frame_idx]

        intrinsic = cam_frame.intrinsic
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        cx = float(intrinsic[0, 2])
        cy = float(intrinsic[1, 2])

        def lift_points(uv: np.ndarray) -> np.ndarray:
            if depth_map is None or uv is None:
                return np.empty((0, 3), dtype=np.float32)
            uv = np.asarray(uv, dtype=np.float32)
            if uv.ndim == 1:
                uv = uv[None, :]
            u = uv[:, 0]
            v = uv[:, 1]
            if depth_scale_u is not None and depth_scale_v is not None:
                u_depth = u * depth_scale_u
                v_depth = v * depth_scale_v
            else:
                u_depth = u
                v_depth = v
            uu = np.round(u_depth).astype(int)
            vv = np.round(v_depth).astype(int)
            depth_h, depth_w = depth_map.shape[:2]
            valid = (
                (uu >= 0)
                & (uu < depth_w)
                & (vv >= 0)
                & (vv < depth_h)
            )
            points = np.zeros((uv.shape[0], 3), dtype=np.float32)
            points[:] = np.nan
            valid_idx = np.where(valid)[0]
            for idx_valid in valid_idx:
                z = float(depth_map[vv[idx_valid], uu[idx_valid]])
                if z <= 0:
                    continue
                x = (u[idx_valid] - cx) * z / fx
                y = (v[idx_valid] - cy) * z / fy
                points[idx_valid] = (x, y, z)
            return points

        # Lift wrist/keypoints using depth when possible
        wrist_cam = None
        keypoints_cam = None
        # import pdb;pdb.set_trace()
        if hand_frame.wrist_uv is not None and hand_frame.keypoints_uv is not None:
            lifted_wrist = lift_points(hand_frame.wrist_uv)
            if lifted_wrist.size and not np.isnan(lifted_wrist).any():
                wrist_cam = lifted_wrist[0]
            lifted_keypoints = lift_points(hand_frame.keypoints_uv)
            if lifted_keypoints.size:
                keypoints_cam = lifted_keypoints

        if wrist_cam is None or np.isnan(wrist_cam).any():
            wrist_cam = hand_frame.wrist_position.astype(np.float32)

        fallback_local = hand_frame.keypoints.astype(np.float32)
        if hand_frame.cam_translation is not None:
            cam_t = hand_frame.cam_translation.astype(np.float32)
        else:
            cam_t = np.zeros(3, dtype=np.float32)
        wrist_quat_cam = hand_frame.wrist_quaternion.astype(np.float32)
        R_hand_cam = rotations.matrix_from_quaternion(wrist_quat_cam)
        fallback_keypoints = (R_hand_cam @ fallback_local.T).T + cam_t[None, :]
        keypoints_cam = fallback_keypoints

        R_wc = cam_frame.cam_c2w[:3, :3]
        t_wc = cam_frame.cam_c2w[:3, 3]

        keypoints_cam0 = (R_wc @ keypoints_cam.T).T + t_wc
        wrist_cam0 = R_wc @ wrist_cam + t_wc

        R_hand_cam = rotations.matrix_from_quaternion(hand_frame.wrist_quaternion)
        R_hand_cam0 = R_wc @ R_hand_cam
        wrist_quat_cam0 = rotations.quaternion_from_matrix(R_hand_cam0)

        aligned_frames.append(
            AlignedHandFrame(
                frame_idx=hand_frame.frame_idx,
                timestamp=hand_frame.timestamp,
                hand_type=hand_frame.hand_type,
                confidence=hand_frame.confidence,
                wrist_cam=wrist_cam.astype(np.float32),
                wrist_cam0=wrist_cam0.astype(np.float32),
                wrist_quat_cam=hand_frame.wrist_quaternion.astype(np.float32),
                wrist_quat_cam0=wrist_quat_cam0.astype(np.float32),
                keypoints_cam=keypoints_cam.astype(np.float32),
                keypoints_cam0=keypoints_cam0.astype(np.float32),
            )
        )

    return AlignedSequence(frames=aligned_frames)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Align HAMER outputs using MegaSaM intrinsics/extrinsics")
    parser.add_argument("hamer_json", type=Path, help="Path to HAMER sequence JSON")
    parser.add_argument("megasam_npz", type=Path, help="MegaSaM *_droid.npz result")
    parser.add_argument("--output", type=Path, default=Path("video2real_outputs/aligned.json"))
    args = parser.parse_args(argv)

    hamer_seq = HAMERSequenceResult.from_json(args.hamer_json)
    cameras = load_megasam_solution(args.megasam_npz)
    aligned = align_hamer_to_cam0(hamer_seq, cameras)
    aligned.to_json(args.output)
    print(f"Aligned {len(aligned.frames)} hand frames to cam0 space: {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
