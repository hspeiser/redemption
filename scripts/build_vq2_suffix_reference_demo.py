"""Convert a saved suffix reference into the live SAC demo schema.

``fastsim_suffix_covis.py`` stores the optimized 30 Hz reference states
directly (position, velocity, attitude, and target gate).  The live teacher
and the full-course ensemble audit consume the SAC demonstration schema
instead.  This adapter preserves the saved geometry/timing and reconstructs
only the feed-forward rates/thrust and deployable observations required by
``build_vq2_hybrid_suffix_demo.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.lineopt import (  # noqa: E402
    DT,
    G,
    RATE_ACTUAL,
    _smooth,
)
from aigp.fastsim.sysid import SurrogateModel  # noqa: E402
from aigp.rl.vq2_features import (  # noqa: E402
    build_observation,
    course_progress,
    load_gate_geometry,
)


def reconstruct_feedforward(
    velocity: np.ndarray,
    quat_wxyz: np.ndarray,
    model: SurrogateModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Return policy actions and body rates for saved reference states."""
    rotation = Rotation.from_quat(np.roll(quat_wxyz, -1, axis=1))
    delta = (rotation[:-1].inv() * rotation[1:]).as_rotvec() / DT
    rates = np.vstack([delta, delta[-1:]])

    velocity_smooth = _smooth(velocity, 5)
    acceleration = _smooth(np.gradient(velocity_smooth, DT, axis=0), 5)
    acceleration = np.clip(acceleration, -25.0, 25.0)
    acceleration[:, 2] = np.minimum(acceleration[:, 2], 0.5 * G)
    thrust_vector = acceleration - np.array([0.0, 0.0, G])
    body_z = rotation.as_matrix()[:, :, 2]
    thrust_acc = np.maximum(-(thrust_vector * body_z).sum(axis=1), 3.0)

    g1 = float(model.thrust_gain)
    g2 = float(model.thrust_quad)
    wire = (-g1 + np.sqrt(g1 * g1 + 4.0 * g2 * thrust_acc)) / (2.0 * g2)
    actions = np.empty((len(velocity), 4), np.float32)
    actions[:, :3] = np.clip(rates / RATE_ACTUAL, -1.0, 1.0)
    actions[:, 3] = 2.0 * np.clip(wire, 0.02, 0.52) - 1.0
    return actions, rates


def build_suffix_demo(
    reference_path: Path,
    map_path: Path,
    model_path: Path,
) -> dict[str, np.ndarray]:
    saved = np.load(reference_path, allow_pickle=False)
    required = {"ref_pos", "ref_vel", "ref_quat", "ref_gate"}
    missing = required.difference(saved.files)
    if missing:
        raise ValueError(f"suffix reference is missing keys: {sorted(missing)}")

    position = np.asarray(saved["ref_pos"], np.float64)
    velocity = np.asarray(saved["ref_vel"], np.float64)
    quat = np.asarray(saved["ref_quat"], np.float64)
    gate_raw = np.asarray(saved["ref_gate"], np.int64)
    n = len(position)
    if not (len(velocity) == len(quat) == len(gate_raw) == n and n >= 3):
        raise ValueError("suffix reference arrays have inconsistent lengths")
    if np.any(np.diff(gate_raw) < 0):
        raise ValueError("suffix target gates must be monotonic")
    gate = np.clip(gate_raw, 0, 16)

    quat_norm = np.linalg.norm(quat, axis=1)
    if not np.allclose(quat_norm, 1.0, atol=2e-3):
        raise ValueError("suffix reference contains non-unit quaternions")

    model = SurrogateModel.load(model_path)
    action, rates = reconstruct_feedforward(velocity, quat, model)
    geometry = load_gate_geometry(json.loads(map_path.read_text())["gates"])

    observation = np.zeros((n, 53), np.float32)
    previous_action = np.zeros(4, np.float32)
    for index in range(n):
        observation[index] = build_observation(
            position_world=position[index],
            quat_wxyz=quat[index],
            velocity_world=velocity[index],
            gyro_raw=-rates[index],
            previous_action=previous_action,
            gate_index=int(gate[index]),
            geometry=geometry,
            position_sigma_m=0.03,
        )
        previous_action = action[index]

    reward = np.zeros(n, np.float32)
    previous_progress = course_progress(
        position[0], int(gate[0]), geometry, position[0]
    )
    previous_action = np.zeros(4, np.float32)
    for index in range(n):
        progress = course_progress(
            position[index], int(gate[index]), geometry, position[0]
        )
        passed = index > 0 and gate_raw[index] > gate_raw[index - 1]
        reward[index] = (
            2.0 * np.clip(progress - previous_progress, -0.5, 1.0)
            + 25.0 * passed
            - 0.8 * DT
            - 0.02 * np.sum((action[index] - previous_action) ** 2)
        )
        previous_progress = progress
        previous_action = action[index]
    reward[-1] += 600.0
    done = np.zeros(n, np.float32)
    done[-1] = 1.0

    return {
        "observation": observation,
        "action": action,
        "reward": reward,
        "next_observation": np.vstack([observation[1:], observation[-1:]]),
        "done": done,
        "wall": 1785700000.0 + np.arange(n) * DT,
        "gate_index": gate,
        "position": position,
        "velocity": velocity,
        "sigma": np.full(n, 0.03, np.float64),
        "source_episode": np.str_(f"suffix-reference/{reference_path.stem}"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--map",
        type=Path,
        default=REPO / "data/vq2_runtime_map_g9g15fix.json",
    )
    parser.add_argument(
        "--model", type=Path, default=REPO / "data/fastsim_model_v2.json"
    )
    args = parser.parse_args()

    payload = build_suffix_demo(args.reference, args.map, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)
    speed = np.linalg.norm(payload["velocity"], axis=1)
    print(json.dumps({
        "out": str(args.out.resolve()),
        "rows": int(len(payload["gate_index"])),
        "suffix_s": float(len(payload["gate_index"]) * DT),
        "gate_min": int(payload["gate_index"].min()),
        "gate_max": int(payload["gate_index"].max()),
        "peak_speed_mps": float(speed.max()),
        "peak_action_abs": float(np.abs(payload["action"]).max()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
