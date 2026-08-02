"""Blend a proven demo toward a faster reference by gate-local phase.

The first N gate segments are resampled to an interpolated duration, then
position, velocity, attitude, rate, and action are blended at equal local
course phase. The untouched suffix comes from the proven base demo.
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

from aigp.rl.vq2_features import (  # noqa: E402
    MAX_RATE,
    build_observation,
    load_gate_geometry,
)


def quaternion_from_observation(observation: np.ndarray) -> np.ndarray:
    first = np.asarray(observation[21:24], float)
    first /= np.linalg.norm(first) + 1e-9
    second = np.asarray(observation[24:27], float)
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second) + 1e-9
    matrix = np.column_stack([first, second, np.cross(first, second)])
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1)


def interp_rows(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values)
    old = np.linspace(0.0, 1.0, len(values))
    new = np.linspace(0.0, 1.0, count)
    if values.ndim == 1:
        return np.interp(new, old, values)
    return np.stack([
        np.interp(new, old, values[:, axis])
        for axis in range(values.shape[1])
    ], axis=1)


def nlerp(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    b = b.copy()
    b[np.sum(a * b, axis=1) < 0.0] *= -1.0
    q = (1.0 - alpha) * a + alpha * b
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--blend-gates", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1]")

    base = np.load(args.base, allow_pickle=False)
    target = np.load(args.target, allow_pickle=False)
    geometry = load_gate_geometry(json.loads(args.map.read_text())["gates"])
    fields = {key: [] for key in ("position", "velocity", "action", "quat", "gyro", "sigma", "gate")}
    segment_counts = []

    def append_segment(source, rows: np.ndarray) -> None:
        fields["position"].append(np.asarray(source["position"])[rows])
        fields["velocity"].append(np.asarray(source["velocity"])[rows])
        fields["action"].append(np.asarray(source["action"])[rows])
        observation = np.asarray(source["observation"])[rows]
        fields["quat"].append(np.asarray([
            quaternion_from_observation(row) for row in observation
        ]))
        fields["gyro"].append(-observation[:, 27:30] * MAX_RATE)
        fields["sigma"].append(observation[:, 51] * 0.5)
        fields["gate"].append(np.asarray(source["gate_index"])[rows])

    for gate in range(args.blend_gates):
        base_rows = np.flatnonzero(np.asarray(base["gate_index"]) == gate)
        target_rows = np.flatnonzero(np.asarray(target["gate_index"]) == gate)
        if not len(base_rows) or not len(target_rows):
            raise ValueError(f"gate {gate} absent from one reference")
        count = max(2, int(round(
            (1.0 - args.alpha) * len(base_rows)
            + args.alpha * len(target_rows)
        )))
        base_position = interp_rows(base["position"][base_rows], count)
        target_position = interp_rows(target["position"][target_rows], count)
        base_velocity = interp_rows(base["velocity"][base_rows], count)
        target_velocity = interp_rows(target["velocity"][target_rows], count)
        base_action = interp_rows(base["action"][base_rows], count)
        target_action = interp_rows(target["action"][target_rows], count)
        base_observation = np.asarray(base["observation"])[base_rows]
        target_observation = np.asarray(target["observation"])[target_rows]
        base_quat = interp_rows(np.asarray([
            quaternion_from_observation(row) for row in base_observation
        ]), count)
        target_quat = interp_rows(np.asarray([
            quaternion_from_observation(row) for row in target_observation
        ]), count)
        base_quat /= np.linalg.norm(base_quat, axis=1, keepdims=True)
        target_quat /= np.linalg.norm(target_quat, axis=1, keepdims=True)
        base_gyro = interp_rows(-base_observation[:, 27:30] * MAX_RATE, count)
        target_gyro = interp_rows(-target_observation[:, 27:30] * MAX_RATE, count)
        base_sigma = interp_rows(base_observation[:, 51] * 0.5, count)
        target_sigma = interp_rows(target_observation[:, 51] * 0.5, count)
        alpha = float(args.alpha)
        fields["position"].append((1 - alpha) * base_position + alpha * target_position)
        fields["velocity"].append((1 - alpha) * base_velocity + alpha * target_velocity)
        fields["action"].append((1 - alpha) * base_action + alpha * target_action)
        fields["quat"].append(nlerp(base_quat, target_quat, alpha))
        fields["gyro"].append((1 - alpha) * base_gyro + alpha * target_gyro)
        fields["sigma"].append((1 - alpha) * base_sigma + alpha * target_sigma)
        fields["gate"].append(np.full(count, gate, np.int64))
        segment_counts.append({
            "gate": gate,
            "base_rows": len(base_rows),
            "target_rows": len(target_rows),
            "output_rows": count,
        })

    suffix = np.flatnonzero(np.asarray(base["gate_index"]) >= args.blend_gates)
    append_segment(base, suffix)
    position = np.concatenate(fields["position"])
    velocity = np.concatenate(fields["velocity"])
    action = np.clip(np.concatenate(fields["action"]), -1.0, 1.0).astype(np.float32)
    quat = np.concatenate(fields["quat"])
    gyro = np.concatenate(fields["gyro"])
    sigma = np.concatenate(fields["sigma"])
    gate_index = np.concatenate(fields["gate"]).astype(np.int64)
    observation = np.zeros((len(position), 53), np.float32)
    previous_action = np.zeros(4, np.float32)
    for row in range(len(position)):
        observation[row] = build_observation(
            position_world=position[row],
            quat_wxyz=quat[row],
            velocity_world=velocity[row],
            gyro_raw=gyro[row],
            previous_action=previous_action,
            gate_index=int(gate_index[row]),
            geometry=geometry,
            position_sigma_m=float(sigma[row]),
        )
        previous_action = action[row]
    wall_start = float(np.asarray(base["wall"])[0])
    wall = wall_start + np.arange(len(position)) / 30.0
    done = np.zeros(len(position), np.float32)
    done[-1] = 1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        observation=observation,
        next_observation=np.vstack([observation[1:], observation[-1:]]),
        action=action,
        reward=np.zeros(len(position), np.float32),
        done=done,
        wall=wall,
        gate_index=gate_index,
        position=position.astype(np.float64),
        velocity=velocity.astype(np.float64),
        sigma=sigma.astype(np.float64),
        source_episode=np.str_(
            f"phase blend alpha={args.alpha:.3f}: {args.base.name} -> {args.target.name}"
        ),
    )
    first_suffix = np.flatnonzero(gate_index >= args.blend_gates)
    gate4_time = (
        float(first_suffix[0]) / 30.0 if len(first_suffix) else None
    )
    print(json.dumps({
        "out": str(args.out),
        "alpha": args.alpha,
        "gate4_time_s": gate4_time,
        "rows": len(position),
        "segments": segment_counts,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
