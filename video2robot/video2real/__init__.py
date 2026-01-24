"""Tools for converting video hand motion to Inspire robot trajectories."""

from .camera_estimation import (
    CameraFrame,
    MegaSamCameraSolution,
    estimate_cameras_from_video,
    load_megasam_solution,
)
from .hamer_extractor import HAMERFrameResult, HAMERSequenceResult, HAMERExtractor
from .cam_alignment import AlignedSequence, AlignedHandFrame, align_hamer_to_cam0
from .wrist_frame import (
    WristPoseG1,
    cam_to_g1_position,
    cam_to_g1_rotation,
    cam_quat_to_g1,
    convert_sequence_wrist_poses,
)
from .run_pipeline import run_pipeline

__all__ = [
    "MegaSamCameraSolution",
    "CameraFrame",
    "HAMERFrameResult",
    "HAMERSequenceResult",
    "HAMERExtractor",
    "align_hamer_to_cam0",
    "AlignedSequence",
    "AlignedHandFrame",
    "cam_to_g1_position",
    "cam_to_g1_rotation",
    "cam_quat_to_g1",
    "convert_sequence_wrist_poses",
    "WristPoseG1",
    "estimate_cameras_from_video",
    "load_megasam_solution",
    "run_pipeline",
]
