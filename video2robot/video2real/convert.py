"""Convert Inspire joint angles to DOF motor commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np


def to_motor_cmd(q_target: float, min_val: float, max_val: float) -> float:
    """Clamp and normalize joint value into [0, 1]."""

    return float(np.clip((max_val - q_target) / (max_val - min_val), 0.0, 1.0))


def convert_qpos_to_dof(qpos_list: Iterable[Sequence[float]]) -> List[List[float]]:
    """Map Inspire joint angles to six motor DOFs."""

    dof_values: List[List[float]] = []
    for joint in qpos_list:
        jq = np.asarray(joint, dtype=np.float32)
        if jq.shape[0] < 10:
            raise ValueError("Expected at least 10 joint values per frame")
        dof = np.zeros(6, dtype=np.float32)
        dof[0] = to_motor_cmd(float(jq[4]), 0.0, 1.7)  # pinky
        dof[1] = to_motor_cmd(float(jq[6]), 0.0, 1.7)  # ring
        dof[2] = to_motor_cmd(float(jq[2]), 0.0, 1.7)  # middle
        dof[3] = to_motor_cmd(float(jq[0]), 0.0, 1.7)  # index
        dof[4] = to_motor_cmd(float(jq[9]), 0.0, 0.5)  # thumb bend
        dof[5] = to_motor_cmd(float(jq[8]), 0.0, 1.3)  # thumb rotation
        dof_values.append(dof.tolist())
    return dof_values


def save_cpp_pushback_lines(dof_list: Iterable[Sequence[float]], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for dof in dof_list:
            line = ", ".join(f"{float(v):.6f}" for v in dof)
            f.write(line + "\n")


def main() -> None:  # pragma: no cover - CLI entry point
    parser = argparse.ArgumentParser(description="Convert Inspire joint angles to DOF commands")
    parser.add_argument("input", type=Path, help="Input JSON with joint angles")
    parser.add_argument("--output", type=Path, default=Path("cmd.json"))
    parser.add_argument("--cpp", type=Path, default=None)
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    qpos_list = data.get("qpos") or [frame["joint_angles"] for frame in data.get("frames", [])]
    if not qpos_list:
        raise ValueError("No joint angles found in the input file")

    dof_list = convert_qpos_to_dof(qpos_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"cmd": dof_list}, indent=2))

    if args.cpp is not None:
        save_cpp_pushback_lines(dof_list, args.cpp)

    print(f"Converted {len(dof_list)} frames to DOF commands -> {args.output}")
    if args.cpp:
        print(f"C++ push_back lines saved to {args.cpp}")


if __name__ == "__main__":
    import argparse

    main()
