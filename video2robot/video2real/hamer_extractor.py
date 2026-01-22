"""HAMER-based hand pose extraction utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
from pytransform3d import rotations

from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

from vitpose_model import ViTPoseModel


@dataclass
class HAMERFrameResult:
    """Per-hand HAMER inference output for one frame."""

    frame_idx: int
    timestamp: float
    hand_type: str
    confidence: float
    wrist_position: np.ndarray  # shape (3,)
    wrist_quaternion: np.ndarray  # wxyz
    keypoints: np.ndarray  # shape (21, 3)
    wrist_uv: Optional[np.ndarray] = None  # shape (2,)
    keypoints_uv: Optional[np.ndarray] = None  # shape (21, 2)
    cam_translation: Optional[np.ndarray] = None  # shape (3,)
    vertices: Optional[np.ndarray] = None  # (778, 3)

    def to_dict(self) -> Dict:
        payload = asdict(self)
        payload["wrist_position"] = self.wrist_position.tolist()
        payload["wrist_quaternion"] = self.wrist_quaternion.tolist()
        payload["keypoints"] = self.keypoints.tolist()
        if self.wrist_uv is not None:
            payload["wrist_uv"] = self.wrist_uv.tolist()
        if self.keypoints_uv is not None:
            payload["keypoints_uv"] = self.keypoints_uv.tolist()
        if self.cam_translation is not None:
            payload["cam_translation"] = self.cam_translation.tolist()
        if self.vertices is not None:
            payload["vertices"] = self.vertices.tolist()
        return payload


@dataclass
class HAMERSequenceResult:
    """Aggregated HAMER results across a sequence."""

    video_path: Path
    frames: List[HAMERFrameResult]
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    frame_size: Optional[Sequence[int]] = None  # (width, height)
    intrinsic: Optional[np.ndarray] = None  # 3x3 camera intrinsic

    def filter_by_hand(self, hand: str) -> List[HAMERFrameResult]:
        return [frame for frame in self.frames if frame.hand_type.lower() == hand.lower()]

    def to_json(self, output_path: Path) -> None:
        payload = {
            "video_path": str(self.video_path),
            "frames": [frame.to_dict() for frame in self.frames],
        }
        if self.fps is not None:
            payload["fps"] = float(self.fps)
        if self.total_frames is not None:
            payload["total_frames"] = int(self.total_frames)
        if self.frame_size is not None:
            payload["frame_size"] = [int(self.frame_size[0]), int(self.frame_size[1])]
        if self.intrinsic is not None:
            payload["intrinsic"] = self.intrinsic.tolist()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def from_json(cls, json_path: Path) -> "HAMERSequenceResult":
        data = json.loads(json_path.read_text())
        frames = []
        for item in data["frames"]:
            frames.append(
                HAMERFrameResult(
                    frame_idx=item["frame_idx"],
                    timestamp=item["timestamp"],
                    hand_type=item["hand_type"],
                    confidence=item["confidence"],
                    wrist_position=np.asarray(item["wrist_position"], dtype=np.float32),
                    wrist_quaternion=np.asarray(item.get("wrist_quaternion", [1.0, 0.0, 0.0, 0.0]), dtype=np.float32),
                    keypoints=np.asarray(item["keypoints"], dtype=np.float32),
                    wrist_uv=np.asarray(item["wrist_uv"], dtype=np.float32) if "wrist_uv" in item else None,
                    keypoints_uv=np.asarray(item["keypoints_uv"], dtype=np.float32) if "keypoints_uv" in item else None,
                    cam_translation=
                    np.asarray(item["cam_translation"], dtype=np.float32)
                    if "cam_translation" in item
                    else None,
                    vertices=
                    np.asarray(item["vertices"], dtype=np.float32)
                    if "vertices" in item
                    else None,
                )
            )
        return cls(
            video_path=Path(data["video_path"]),
            frames=frames,
            fps=data.get("fps"),
            total_frames=data.get("total_frames"),
            frame_size=data.get("frame_size"),
            intrinsic=np.asarray(data["intrinsic"], dtype=np.float32) if "intrinsic" in data else None,
        )


class HAMERExtractor:
    """Headless HAMER video processor without the legacy ``video_demo`` wrapper."""

    def __init__(
        self,
        *,
        checkpoint_path: Optional[str] = None,
        body_detector: str = "vitdet",
        rescale_factor: float = 2.0,
        batch_size: int = 8,
        store_vertices: bool = False,
        mesh_dir: Optional[Path] = None,
        full_focal_length: Optional[float] = None,
        intrinsic_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = checkpoint_path or DEFAULT_CHECKPOINT
        self.model, self.model_cfg = load_hamer(ckpt)
        self.model = self.model.to(self.device)
        self.model.eval()
        self.joint_regressor = self.model.mano.J_regressor.to(self.device)

        self.rescale_factor = rescale_factor
        self.batch_size = max(1, batch_size)
        self.detector = self._build_detector(body_detector)
        self.cpm = ViTPoseModel(self.device)
        self.store_vertices = store_vertices
        self.mesh_dir: Optional[Path] = None
        if mesh_dir is not None:
            mesh_dir = Path(mesh_dir)
            mesh_dir.mkdir(parents=True, exist_ok=True)
            self.mesh_dir = mesh_dir
        self.renderer: Optional[Renderer] = None
        if self.store_vertices or self.mesh_dir is not None:
            self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)
        self._focal_length_override: Optional[float] = (
            float(full_focal_length) if full_focal_length is not None else None
        )
        self.intrinsic_matrix: Optional[np.ndarray]
        if intrinsic_matrix is not None:
            matrix = np.asarray(intrinsic_matrix, dtype=np.float32)
            if matrix.shape != (3, 3):
                raise ValueError("intrinsic_matrix must be a 3x3 matrix")
            self.intrinsic_matrix = matrix
            if self._focal_length_override is None:
                self._focal_length_override = float(matrix[0, 0])
        else:
            self.intrinsic_matrix = None
        if self.intrinsic_matrix is not None:
            self._proj_params: Optional[tuple[float, float, float, float]] = (
                float(self.intrinsic_matrix[0, 0]),
                float(self.intrinsic_matrix[1, 1]),
                float(self.intrinsic_matrix[0, 2]),
                float(self.intrinsic_matrix[1, 2]),
            )
        else:
            self._proj_params = None

    @staticmethod
    def _empty_frame_result() -> Dict[str, List]:
        return {
            "hands": [],
            "wrist_positions": [],
            "wrist_orientations": [],
            "keypoints": [],
            "wrist_uvs": [],
            "keypoints_uvs": [],
            "vertices": [],
            "cam_translations": [],
        }

    def _build_detector(self, detector_type: str):
        from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

        detector_type = detector_type.lower()
        if detector_type == "vitdet":
            from detectron2.config import LazyConfig
            import hamer

            cfg_path = Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = (
                "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
                "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            )
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            return DefaultPredictor_Lazy(detectron2_cfg)

        if detector_type == "regnety":
            from detectron2 import model_zoo

            detectron2_cfg = model_zoo.get_config(
                "new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py", trained=True
            )
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            return DefaultPredictor_Lazy(detectron2_cfg)

        raise ValueError(f"Unsupported detector type: {detector_type}")

    def _project_points(self, points_cam: np.ndarray) -> np.ndarray:
        if self._proj_params is None:
            raise ValueError("Projection parameters unavailable; provide intrinsic_matrix or --megasam-npz")
        fx, fy, cx, cy = self._proj_params
        pts = np.asarray(points_cam, dtype=np.float32)
        uv = np.full((pts.shape[0], 2), np.nan, dtype=np.float32)
        z = pts[:, 2]
        valid = np.abs(z) > 1e-6
        if np.any(valid):
            uv[valid, 0] = fx * pts[valid, 0] / z[valid] + cx
            uv[valid, 1] = fy * pts[valid, 1] / z[valid] + cy
        return uv

    def _process_frame(self, frame: np.ndarray) -> Dict[str, List]:
        det_out = self.detector(frame)
        img_rgb = frame[:, :, ::-1]

        instances = det_out["instances"]
        valid_idx = (instances.pred_classes == 0) & (instances.scores > 0.5)
        pred_bboxes = instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = instances.scores[valid_idx].cpu().numpy()

        if pred_bboxes.shape[0] == 0:
            return self._empty_frame_result()

        vitpose_inputs = [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)]
        vitposes_out = self.cpm.predict_pose(img_rgb, vitpose_inputs)

        hand_entries: List[Dict[str, object]] = []
        for det_idx, vitposes in enumerate(vitposes_out):
            score = float(pred_scores[det_idx]) if det_idx < len(pred_scores) else 1.0
            keypoints = vitposes["keypoints"]
            left_hand_keyp = keypoints[-42:-21]
            right_hand_keyp = keypoints[-21:]
            for kp_set, is_right in ((left_hand_keyp, False), (right_hand_keyp, True)):
                valid = kp_set[:, 2] > 0.5
                if np.sum(valid) <= 3:
                    continue
                bbox = [
                    float(kp_set[valid, 0].min()),
                    float(kp_set[valid, 1].min()),
                    float(kp_set[valid, 0].max()),
                    float(kp_set[valid, 1].max()),
                ]
                hand_entries.append(
                    {
                        "bbox": bbox,
                        "is_right": is_right,
                        "score": score,
                        "keypoints_uv": kp_set.copy(),
                    }
                )

        if not hand_entries:
            return self._empty_frame_result()

        boxes = np.stack([entry["bbox"] for entry in hand_entries])
        right_flags = np.array([1 if entry["is_right"] else 0 for entry in hand_entries], dtype=np.int32)
        scores = np.array([entry["score"] for entry in hand_entries], dtype=np.float32)

        dataset = ViTDetDataset(
            self.model_cfg,
            frame,
            boxes,
            right_flags,
            rescale_factor=self.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        all_hands: List[Dict[str, object]] = []
        all_wrist_positions: List[List[float]] = []
        all_wrist_orientations: List[List[float]] = []
        all_keypoints: List[List[List[float]]] = []
        all_wrist_uvs: List[List[float]] = []
        all_keypoints_uvs: List[List[List[float]]] = []
        all_vertices: List[List[List[float]]] = []
        all_cam_translations: List[List[float]] = []

        hand_counter = 0
        flip_mat = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]

            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            if self._focal_length_override is not None:
                focal_length = torch.as_tensor(
                    self._focal_length_override,
                    dtype=pred_cam.dtype,
                    device=pred_cam.device,
                )
            else:
                focal_length = (
                    torch.as_tensor(
                        float(self.model_cfg.EXTRA.FOCAL_LENGTH),
                        dtype=pred_cam.dtype,
                        device=pred_cam.device,
                    )
                    / float(self.model_cfg.MODEL.IMAGE_SIZE)
                    * img_size.max()
                )

            pred_cam_t_full = cam_crop_to_full(
                pred_cam,
                box_center,
                box_size,
                img_size,
                focal_length,
            ).detach().cpu().numpy()

            global_orients = out["pred_mano_params"]["global_orient"].detach().cpu().numpy()
            pred_vertices = out["pred_vertices"].detach().cpu().numpy()
            if "pred_keypoints_3d" in out:
                pred_keypoints_local = out["pred_keypoints_3d"].detach().cpu().numpy()
            else:
                pred_keypoints_local = pred_vertices[:, :21]

            batch_size = pred_vertices.shape[0]
            for n in range(batch_size):
                idx = hand_counter + n
                is_right_hand = bool(right_flags[idx])

                verts = pred_vertices[n].copy()
                if not is_right_hand:
                    verts[:, 0] *= -1.0

                cam_t = pred_cam_t_full[n]

                keypoints_local = pred_keypoints_local[n].copy()
                if not is_right_hand:
                    keypoints_local[:, 0] *= -1.0

                # Lift MANO joint positions into the camera frame
                joints_3d = (
                    torch.matmul(self.joint_regressor, out["pred_vertices"][n])
                    .detach()
                    .cpu()
                    .numpy()
                )
                handedness_scalar = 1.0 if is_right_hand else -1.0
                joints_3d[:, 0] *= handedness_scalar
                joints_cam = joints_3d + cam_t[None, :]
                wrist_pos = joints_cam[0]

                keypoints_cam = keypoints_local + cam_t[None, :]

                entry_meta = hand_entries[idx]
                kp_uv_full = entry_meta["keypoints_uv"]
                projected_keypoints_uv: Optional[np.ndarray] = None
                projected_wrist_uv: Optional[np.ndarray] = None
                if self._proj_params is not None:
                    projected_keypoints_uv = self._project_points(keypoints_cam)
                    fallback_uv = kp_uv_full[:, :2].astype(np.float32)
                    invalid = np.isnan(projected_keypoints_uv).any(axis=1)
                    if invalid.any():
                        projected_keypoints_uv[invalid] = fallback_uv[invalid]
                    wrist_proj = self._project_points(joints_cam[[0]])[0]
                    if np.isnan(wrist_proj).any():
                        wrist_proj = fallback_uv[0]
                    projected_wrist_uv = wrist_proj.astype(np.float32)
                wrist_uv = (
                    projected_wrist_uv
                    if projected_wrist_uv is not None
                    else kp_uv_full[0, :2].astype(np.float32)
                )
                keypoints_uv = (
                    projected_keypoints_uv
                    if projected_keypoints_uv is not None
                    else kp_uv_full[:, :2].astype(np.float32)
                )

                orient_mat = global_orients[n]
                if orient_mat.ndim == 3:
                    orient_mat = orient_mat[0]
                orient_mat = orient_mat.astype(np.float32)
                if not is_right_hand:
                    orient_mat = flip_mat @ orient_mat @ flip_mat
                wrist_quat = rotations.quaternion_from_matrix(orient_mat).astype(np.float32)

                hand_data = {
                    "hand_type": "right" if is_right_hand else "left",
                    "is_right": is_right_hand,
                    "bbox": boxes[idx].tolist(),
                    "confidence": float(scores[idx]),
                }

                all_hands.append(hand_data)
                all_wrist_positions.append(wrist_pos.tolist())
                all_wrist_orientations.append(wrist_quat.tolist())
                all_keypoints.append(keypoints_local.tolist())
                all_wrist_uvs.append(wrist_uv.astype(np.float32).tolist())
                all_keypoints_uvs.append(keypoints_uv.astype(np.float32).tolist())
                if self.store_vertices:
                    all_vertices.append(verts.tolist())
                all_cam_translations.append(cam_t.tolist())

            hand_counter += batch_size

        return {
            "hands": all_hands,
            "wrist_positions": all_wrist_positions,
            "wrist_orientations": all_wrist_orientations,
            "keypoints": all_keypoints,
            "wrist_uvs": all_wrist_uvs,
            "keypoints_uvs": all_keypoints_uvs,
            **({"vertices": all_vertices} if self.store_vertices else {}),
            "cam_translations": all_cam_translations,
        }

    def run(self, video_path: Path) -> HAMERSequenceResult:
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise OSError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or np.isnan(fps):
            fps = 30.0

        frame_idx = 0
        sequence: List[HAMERFrameResult] = []

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else None
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else None

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_data = self._process_frame(frame)
                hands = frame_data["hands"]
                if not hands:
                    frame_idx += 1
                    continue

                timestamp = frame_idx / fps if fps > 0 else float(frame_idx)
                wrist_orients = frame_data.get("wrist_orientations", [])
                wrist_uvs = frame_data.get("wrist_uvs", [])
                keypoints_uvs = frame_data.get("keypoints_uvs", [])
                cam_translations = frame_data.get("cam_translations", [])
                vertices_list = frame_data.get("vertices", [])
                for idx, hand in enumerate(hands):
                    kp = np.asarray(frame_data["keypoints"][idx], dtype=np.float32)
                    wrist = np.asarray(frame_data["wrist_positions"][idx], dtype=np.float32)
                    wrist_quat = (
                        np.asarray(wrist_orients[idx], dtype=np.float32)
                        if idx < len(wrist_orients)
                        else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    )
                    cam_t = (
                        np.asarray(cam_translations[idx], dtype=np.float32)
                        if idx < len(cam_translations)
                        else None
                    )
                    verts = (
                        np.asarray(vertices_list[idx], dtype=np.float32)
                        if idx < len(vertices_list)
                        else None
                    )
                    wrist_uv = (
                        np.asarray(wrist_uvs[idx], dtype=np.float32)
                        if idx < len(wrist_uvs)
                        else None
                    )
                    keypoints_uv = (
                        np.asarray(keypoints_uvs[idx], dtype=np.float32)
                        if idx < len(keypoints_uvs)
                        else None
                    )
                    if (
                        self.mesh_dir is not None
                        and self.renderer is not None
                        and verts is not None
                        and cam_t is not None
                    ):
                        side_dir = self.mesh_dir / (
                            "right" if hand["hand_type"].lower() == "right" else "left"
                        )
                        side_dir.mkdir(parents=True, exist_ok=True)
                        mesh = self.renderer.vertices_to_trimesh(
                            verts.copy(),
                            cam_t.copy(),
                            mesh_base_color=(0.65, 0.74, 0.86),
                            is_right=1 if hand["hand_type"].lower() == "right" else 0,
                        )
                        mesh_path = side_dir / f"{frame_idx:06d}_{idx:02d}.obj"
                        mesh.export(mesh_path)
                    sequence.append(
                        HAMERFrameResult(
                            frame_idx=frame_idx,
                            timestamp=timestamp,
                            hand_type=str(hand["hand_type"]),
                            confidence=float(hand["confidence"]),
                            wrist_position=wrist,
                            wrist_quaternion=wrist_quat,
                            keypoints=kp,
                            wrist_uv=wrist_uv,
                            keypoints_uv=keypoints_uv,
                            cam_translation=cam_t,
                            vertices=verts,
                        )
                    )

                frame_idx += 1
        finally:
            cap.release()

        return HAMERSequenceResult(
            video_path=video_path,
            frames=sequence,
            fps=fps,
            total_frames=frame_idx,
            frame_size=(frame_width, frame_height) if frame_width and frame_height else None,
            intrinsic=self.intrinsic_matrix.copy() if self.intrinsic_matrix is not None else None,
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract hand trajectories with HAMER")
    parser.add_argument("video", type=Path, help="Input video path")
    parser.add_argument("--output", type=Path, default=Path("video2real_outputs/hamer.json"))
    parser.add_argument("--checkpoint", type=str, default=None, help="Override HAMER checkpoint path")
    parser.add_argument(
        "--body-detector",
        type=str,
        default="vitdet",
        choices=["vitdet", "regnety"],
        help="Human detector backbone used by HAMER",
    )
    parser.add_argument(
        "--rescale-factor",
        type=float,
        default=2.0,
        help="Bounding box padding factor for HAMER cropping",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for HAMER inference",
    )
    parser.add_argument(
        "--store-vertices",
        action="store_true",
        help="Persist MANO vertices in the JSON output",
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=None,
        help="Optional directory to export per-frame OBJ meshes",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=None,
        help="Explicit focal length override in pixels",
    )
    parser.add_argument(
        "--focal-source",
        choices=("hamer", "megasam"),
        default="hamer",
        help="Fallback focal length source when --focal-length is not set",
    )
    parser.add_argument(
        "--megasam-npz",
        type=Path,
        default=None,
        help="MegaSaM *_droid.npz to harvest camera intrinsics",
    )
    args = parser.parse_args(argv)

    kwargs = {
        "body_detector": args.body_detector,
        "rescale_factor": args.rescale_factor,
        "batch_size": args.batch_size,
    }
    intrinsic_matrix: Optional[np.ndarray] = None
    focal_override: Optional[float] = args.focal_length

    if args.megasam_npz is not None:
        npz_path = Path(args.megasam_npz)
        if not npz_path.exists():
            raise FileNotFoundError(f"MegaSaM npz not found: {npz_path}")
        npz_data = np.load(npz_path)
        if "intrinsic" not in npz_data:
            raise KeyError(f"'intrinsic' not found in MegaSaM npz: {npz_path}")
        intrinsic_raw = npz_data["intrinsic"]
        if intrinsic_raw.ndim == 3:
            intrinsic_matrix = intrinsic_raw[0].astype(np.float32)
        else:
            intrinsic_matrix = intrinsic_raw.astype(np.float32)
        if intrinsic_matrix.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 intrinsic matrix, got shape {intrinsic_matrix.shape} from {npz_path}"
            )
        if focal_override is None and args.focal_source == "megasam":
            focal_override = float(intrinsic_matrix[0, 0])

    if focal_override is None and args.focal_source == "megasam":
        raise ValueError("--focal-source megasam requires either --focal-length or --megasam-npz")

    if args.checkpoint:
        kwargs["checkpoint_path"] = args.checkpoint
    if args.store_vertices:
        kwargs["store_vertices"] = True
    if args.mesh_dir is not None:
        kwargs["mesh_dir"] = args.mesh_dir
    if focal_override is not None:
        kwargs["full_focal_length"] = focal_override
    if intrinsic_matrix is not None:
        kwargs["intrinsic_matrix"] = intrinsic_matrix

    extractor = HAMERExtractor(**kwargs)
    result = extractor.run(args.video)
    result.to_json(args.output)
    left_count = len(result.filter_by_hand("left"))
    right_count = len(result.filter_by_hand("right"))
    total_frames = result.total_frames if result.total_frames is not None else "?"
    print(
        "Saved HAMER sequence with "
        f"{len(result.frames)} hands ({left_count} left / {right_count} right) "
        f"across {total_frames} frames to {args.output}"
    )


if __name__ == "__main__":  # pragma: no cover
    main()
