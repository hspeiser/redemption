"""Build a behavior-cloning/SAC seed from a clean VQ2 demonstration.

The authoritative clean lap contains camera, IMU, official race status, and
the exact commands Henry flew.  A previously generated V11/EKF trace supplies
only deployable visual-inertial pose; no VQ2 odometry is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.vq2_features import (  # noqa: E402
    N_RACE_GATES,
    build_observation,
    course_progress,
    load_gate_geometry,
    wire_command_to_action,
)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def nearest_indices(source_time: np.ndarray, query_time: np.ndarray) \
        -> np.ndarray:
    right = np.searchsorted(source_time, query_time, side="left")
    right = np.clip(right, 0, len(source_time) - 1)
    left = np.clip(right - 1, 0, len(source_time) - 1)
    use_left = np.abs(query_time - source_time[left]) <= \
        np.abs(source_time[right] - query_time)
    return np.where(use_left, left, right)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        type=Path,
        default=REPO / "data" / "ep_rc_20260729_000036",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=REPO / "data" / "vq2_trace_gift_v11sparse10hz_ep13.npz",
    )
    parser.add_argument(
        "--runtime-map",
        type=Path,
        default=REPO / "data" /
        "vq2_runtime_map_gift_v11sparse10hz_ep13.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "data" / "vq2_sac_clean_demo.npz",
    )
    args = parser.parse_args()

    trace = np.load(args.trace, allow_pickle=False)
    map_payload = json.loads(args.runtime_map.read_text())
    geometry = load_gate_geometry(map_payload["gates"])

    frame_rows = read_jsonl(args.episode / "frames.jsonl")
    frame_wall = {int(row["idx"]): float(row["wall"]) for row in frame_rows}
    trace_indices = np.array([
        int(Path(str(path).replace("\\", "/")).stem)
        for path in trace["path"]
    ])
    walls = np.array([frame_wall[index] for index in trace_indices], float)

    race_rows = [
        row for row in read_jsonl(args.episode / "mav.jsonl")
        if row.get("kind") == "race_status"
    ]
    reset_row = next(
        row for row in race_rows
        if row["active_gate"] == 0 and row["sim_boot_ms"] < 100
    )
    race_rows = [
        row for row in race_rows
        if row["wall"] >= reset_row["wall"]
    ]
    race_wall = np.array([row["wall"] for row in race_rows], float)
    race_gate = np.maximum.accumulate(np.array([
        row["active_gate"] for row in race_rows
    ], int))
    gate_at_frame = race_gate[
        np.clip(
            np.searchsorted(race_wall, walls, side="right") - 1,
            0,
            len(race_wall) - 1,
        )
    ]
    race_active_wall = next(
        float(row["wall"]) for row in race_rows
        if row["race_start_ms"] > 0
        and row["sim_boot_ms"] >= row["race_start_ms"]
    )

    command_rows = read_jsonl(args.episode / "cmd.jsonl")
    command_wall = np.array([row["wall"] for row in command_rows], float)
    command_value = np.asarray([row["cmd"] for row in command_rows], float)
    command_index = nearest_indices(command_wall, walls)

    imu_rows = read_jsonl(args.episode / "imu.jsonl")
    imu_wall = np.array([row["wall"] for row in imu_rows], float)
    imu_gyro = np.asarray([row["gyro"] for row in imu_rows], float)
    imu_index = nearest_indices(imu_wall, walls)

    keep = (
        (walls >= race_active_wall)
        & (gate_at_frame >= 0)
        & (gate_at_frame < N_RACE_GATES)
        & (np.abs(command_wall[command_index] - walls) <= 0.05)
        & (np.abs(imu_wall[imu_index] - walls) <= 0.05)
    )
    walls = walls[keep]
    positions = np.asarray(trace["pos"], float)[keep]
    quaternions = np.asarray(trace["quat"], float)[keep]
    sigmas = np.asarray(trace["sigma"], float)[keep]
    gate_indices = gate_at_frame[keep]
    commands = command_value[command_index[keep]]
    gyros = imu_gyro[imu_index[keep]]

    if len(walls) < 100:
        raise RuntimeError(f"only {len(walls)} synchronized demonstration rows")
    velocity = np.gradient(positions, walls, axis=0)
    window = min(15, len(velocity) // 2 * 2 - 1)
    if window >= 5:
        velocity = savgol_filter(
            velocity, window_length=window, polyorder=2, axis=0
        )
    actions = np.asarray([
        wire_command_to_action(command) for command in commands
    ])

    observations = []
    previous_action = np.zeros(4, np.float32)
    for row in range(len(walls)):
        observations.append(build_observation(
            position_world=positions[row],
            quat_wxyz=quaternions[row],
            velocity_world=velocity[row],
            gyro_raw=gyros[row],
            previous_action=previous_action,
            gate_index=int(gate_indices[row]),
            geometry=geometry,
            position_sigma_m=float(sigmas[row]),
        ))
        previous_action = actions[row]
    observations = np.asarray(observations, np.float32)

    spawn_position = np.asarray(trace["pos"][0], float)
    progress = np.array([
        course_progress(position, int(gate), geometry, spawn_position)
        for position, gate in zip(positions, gate_indices)
    ])
    delta_progress = np.clip(np.diff(progress), -0.5, 1.0)
    delta_time = np.clip(np.diff(walls), 0.0, 0.2)
    gate_advance = np.maximum(np.diff(gate_indices), 0)
    action_delta = np.diff(actions, axis=0)
    rewards = (
        2.0 * delta_progress
        + 25.0 * gate_advance
        - 0.8 * delta_time
        - 0.02 * np.sum(action_delta * action_delta, axis=1)
    ).astype(np.float32)
    done = np.zeros(len(rewards), np.float32)
    done[-1] = 1.0
    rewards[-1] += 600.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        observation=observations[:-1],
        action=actions[:-1],
        reward=rewards,
        next_observation=observations[1:],
        done=done,
        wall=walls[:-1],
        gate_index=gate_indices[:-1],
        position=positions[:-1],
        velocity=velocity[:-1],
        sigma=sigmas[:-1],
        source_episode=str(args.episode),
        source_trace=str(args.trace),
        source_map=str(args.runtime_map),
    )
    unique, counts = np.unique(gate_indices, return_counts=True)
    print(f"wrote {len(rewards)} transitions -> {args.output}")
    print(
        f"wall span {walls[-1] - walls[0]:.2f}s, "
        f"gates {int(gate_indices.min())}..{int(gate_indices.max())}, "
        f"sigma median/p90 {np.percentile(sigmas, [50, 90]) * 100}cm"
    )
    print("samples per gate:", dict(zip(unique.tolist(), counts.tolist())))
    print(
        "canonical action p01/p50/p99:",
        np.round(np.percentile(actions, [1, 50, 99], axis=0), 3),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
