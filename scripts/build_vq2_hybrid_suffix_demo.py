"""Splice a proven prefix to a faster suffix with a smooth state bridge.

The prefix remains unchanged through the gate immediately before
``--suffix-start-gate``.  A cubic-Hermite bridge then joins its measured
position/velocity to the faster suffix trajectory while an attitude slerp
keeps body-rate commands bounded.  Prefix observations are copied byte for
byte so an offline A/B has exact parity before the splice; only suffix
observations are rebuilt from the merged deployable state.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.fastsim.lineopt import DT, RATE_ACTUAL
from aigp.rl.vq2_features import (
    MAX_RATE,
    build_observation,
    course_progress,
    load_gate_geometry,
)


def gate_index(data: np.lib.npyio.NpzFile) -> np.ndarray:
    key = "gate_index" if "gate_index" in data else "gate"
    return np.asarray(data[key], np.int64)


def rotations_and_rates(
    observations: np.ndarray,
) -> tuple[Rotation, np.ndarray]:
    first = np.asarray(observations[:, 21:24], np.float64)
    second = np.asarray(observations[:, 24:27], np.float64)
    first /= np.linalg.norm(first, axis=1, keepdims=True) + 1e-12
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=1, keepdims=True) + 1e-12
    third = np.cross(first, second)
    matrix = np.stack([first, second, third], axis=2)
    rates = np.asarray(observations[:, 27:30], np.float64) * MAX_RATE
    return Rotation.from_matrix(matrix), rates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--suffix", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--suffix-start-gate", type=int, default=11)
    parser.add_argument("--bridge-rows", type=int, default=30)
    args = parser.parse_args()

    prefix_npz = np.load(args.prefix, allow_pickle=True)
    suffix_npz = np.load(args.suffix, allow_pickle=True)
    prefix_gate_all = gate_index(prefix_npz)
    suffix_gate_all = gate_index(suffix_npz)
    prefix_keep = prefix_gate_all < args.suffix_start_gate
    suffix_keep = suffix_gate_all >= args.suffix_start_gate
    if not prefix_keep.any() or not suffix_keep.any():
        parser.error("requested splice gate is absent from an input demo")

    prefix_position = np.asarray(prefix_npz["position"][prefix_keep], float)
    prefix_velocity = np.asarray(prefix_npz["velocity"][prefix_keep], float)
    prefix_action = np.asarray(prefix_npz["action"][prefix_keep], np.float32)
    prefix_obs = np.asarray(prefix_npz["observation"][prefix_keep], np.float32)
    prefix_sigma = np.asarray(prefix_npz["sigma"][prefix_keep], np.float64)
    prefix_gate = prefix_gate_all[prefix_keep]
    prefix_rot, prefix_rates = rotations_and_rates(prefix_obs)

    suffix_position = np.asarray(suffix_npz["position"][suffix_keep], float)
    suffix_velocity = np.asarray(suffix_npz["velocity"][suffix_keep], float)
    suffix_action = np.asarray(suffix_npz["action"][suffix_keep], np.float32)
    suffix_obs = np.asarray(suffix_npz["observation"][suffix_keep], np.float32)
    suffix_sigma = np.asarray(suffix_npz["sigma"][suffix_keep], np.float64)
    suffix_gate = suffix_gate_all[suffix_keep]
    suffix_rot, suffix_rates = rotations_and_rates(suffix_obs)

    bridge = int(args.bridge_rows)
    if not 4 <= bridge < len(suffix_position) - 2:
        parser.error("--bridge-rows is outside the usable suffix")
    if np.any(suffix_gate[:bridge + 1] != args.suffix_start_gate):
        parser.error("bridge extends beyond the first suffix gate segment")

    p0 = prefix_position[-1]
    v0 = prefix_velocity[-1]
    p1 = suffix_position[bridge].copy()
    v1 = suffix_velocity[bridge].copy()
    duration = (bridge + 1) * DT
    time = np.arange(1, bridge + 2, dtype=float) * DT
    u = time / duration
    h00 = 2 * u**3 - 3 * u**2 + 1
    h10 = u**3 - 2 * u**2 + u
    h01 = -2 * u**3 + 3 * u**2
    h11 = u**3 - u**2
    suffix_position[:bridge + 1] = (
        h00[:, None] * p0
        + h10[:, None] * duration * v0
        + h01[:, None] * p1
        + h11[:, None] * duration * v1
    )
    dh00 = (6 * u**2 - 6 * u) / duration
    dh10 = 3 * u**2 - 4 * u + 1
    dh01 = (-6 * u**2 + 6 * u) / duration
    dh11 = 3 * u**2 - 2 * u
    suffix_velocity[:bridge + 1] = (
        dh00[:, None] * p0
        + dh10[:, None] * v0
        + dh01[:, None] * p1
        + dh11[:, None] * v1
    )

    end_rot = suffix_rot[bridge]
    attitude_bridge = Slerp(
        [0.0, duration],
        Rotation.concatenate([prefix_rot[-1], end_rot]),
    )(time)
    suffix_quat = np.roll(suffix_rot.as_quat(), 1, axis=1)
    suffix_quat[:bridge + 1] = np.roll(
        attitude_bridge.as_quat(), 1, axis=1
    )

    prefix_quat = np.roll(prefix_rot.as_quat(), 1, axis=1)
    all_quat_for_rates = np.vstack([prefix_quat[-1:], suffix_quat])
    all_rot_for_rates = Rotation.from_quat(np.roll(all_quat_for_rates, -1, axis=1))
    delta = (
        all_rot_for_rates[:-1].inv() * all_rot_for_rates[1:]
    ).as_rotvec() / DT
    suffix_rates[:bridge + 1] = delta[:bridge + 1]
    suffix_action[:bridge + 1, :3] = np.clip(
        suffix_rates[:bridge + 1] / RATE_ACTUAL, -1.0, 1.0
    )
    thrust_start = float(prefix_action[-1, 3])
    thrust_end = float(suffix_action[bridge, 3])
    smooth = u * u * (3.0 - 2.0 * u)
    suffix_action[:bridge + 1, 3] = (
        thrust_start + smooth * (thrust_end - thrust_start)
    )

    position = np.vstack([prefix_position, suffix_position])
    velocity = np.vstack([prefix_velocity, suffix_velocity])
    quat = np.vstack([prefix_quat, suffix_quat])
    rates = np.vstack([prefix_rates, suffix_rates])
    action = np.vstack([prefix_action, suffix_action]).astype(np.float32)
    gates = np.concatenate([prefix_gate, suffix_gate]).astype(np.int64)

    geometry = load_gate_geometry(
        json.loads(args.map.read_text())["gates"]
    )
    prefix_rows = len(prefix_gate)
    observations = np.zeros((len(gates), 53), np.float32)
    observations[:prefix_rows] = prefix_obs
    previous_action = action[prefix_rows - 1]
    for index in range(prefix_rows, len(gates)):
        observations[index] = build_observation(
            position_world=position[index],
            quat_wxyz=quat[index],
            velocity_world=velocity[index],
            gyro_raw=-rates[index],
            previous_action=previous_action,
            gate_index=int(gates[index]),
            geometry=geometry,
            position_sigma_m=0.03,
        )
        previous_action = action[index]

    reward = np.zeros(len(gates), np.float32)
    previous_progress = course_progress(
        position[0], int(gates[0]), geometry, position[0]
    )
    previous_action = np.zeros(4, np.float32)
    for index in range(len(gates)):
        progress = course_progress(
            position[index], int(gates[index]), geometry, position[0]
        )
        passed = index > 0 and gates[index] > gates[index - 1]
        reward[index] = (
            2.0 * np.clip(progress - previous_progress, -0.5, 1.0)
            + 25.0 * passed
            - 0.8 * DT
            - 0.02 * np.sum((action[index] - previous_action) ** 2)
        )
        previous_progress = progress
        previous_action = action[index]
    reward[-1] += 600.0
    done = np.zeros(len(gates), np.float32)
    done[-1] = 1.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        observation=observations,
        action=action,
        reward=reward,
        next_observation=np.vstack([observations[1:], observations[-1:]]),
        done=done,
        wall=1785700000.0 + np.arange(len(gates)) * DT,
        gate_index=gates,
        position=position,
        velocity=velocity,
        sigma=np.concatenate([prefix_sigma, suffix_sigma]),
        source_episode=np.str_(
            f"hybrid prefix={args.prefix.name} suffix={args.suffix.name} "
            f"gate={args.suffix_start_gate} bridge={bridge}"
        ),
    )
    acceleration = np.gradient(velocity, DT, axis=0)
    summary = {
        "out": str(args.out.resolve()),
        "rows": len(gates),
        "nominal_lap_s": len(gates) * DT,
        "prefix_rows": len(prefix_gate),
        "suffix_rows": len(suffix_gate),
        "bridge_rows": bridge,
        "bridge_peak_speed_mps": float(np.linalg.norm(
            suffix_velocity[:bridge + 1], axis=1
        ).max()),
        "bridge_peak_accel_mps2": float(np.linalg.norm(
            acceleration[len(prefix_gate):len(prefix_gate) + bridge + 1],
            axis=1,
        ).max()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
