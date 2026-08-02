"""Convert completed GipsyDanger VQ2 flights into the live SAC demo schema.

The fast local demonstration remains first in the output so the live
controller continues to use it as its reference trajectory.  Completed
GipsyDanger flights are appended as additional successful critic/BC support:
the learner treats demonstration actions as zero residual, so these rows teach
the residual actor to preserve success across a wider state distribution
without making the slower controller the new racing reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aigp.rl.vq2_features import (  # noqa: E402
    N_RACE_GATES,
    build_observation,
    course_progress,
    load_gate_geometry,
    wire_command_to_action,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--runs-root", type=Path, required=True)
    result.add_argument("--base-demo", type=Path, required=True)
    result.add_argument("--map", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--hz", type=float, default=30.0)
    result.add_argument("--progress-scale", type=float, default=2.0)
    result.add_argument("--gate-bonus", type=float, default=25.0)
    result.add_argument("--finish-bonus", type=float, default=600.0)
    result.add_argument("--time-penalty-per-s", type=float, default=3.0)
    result.add_argument("--action-smoothness", type=float, default=0.02)
    result.add_argument("--gamma", type=float, default=0.995)
    result.add_argument("--n-step", type=int, default=12)
    return result


def reward_rows(
    *,
    position: np.ndarray,
    gate_index: np.ndarray,
    action: np.ndarray,
    elapsed_s: np.ndarray,
    geometry,
    progress_scale: float,
    gate_bonus: float,
    finish_bonus: float,
    time_penalty_per_s: float,
    action_smoothness: float,
) -> np.ndarray:
    spawn = np.zeros(3, dtype=float)
    progress = np.asarray([
        course_progress(p, int(g), geometry, spawn)
        for p, g in zip(position, gate_index, strict=True)
    ])
    previous_progress = np.r_[0.0, progress[:-1]]
    delta_progress = np.clip(progress - previous_progress, -0.5, 1.0)
    gate_advance = np.maximum(
        gate_index - np.r_[gate_index[0], gate_index[:-1]], 0
    )
    previous_action = np.vstack([np.zeros(4), action[:-1]])
    reward = (
        progress_scale * delta_progress
        + gate_bonus * gate_advance
        - time_penalty_per_s * elapsed_s
        - action_smoothness
        * np.sum((action - previous_action) ** 2, axis=1)
    )
    reward[-1] += finish_bonus
    return reward.astype(np.float32)


def completed_gate_timeline(result: dict, times: np.ndarray) -> np.ndarray:
    gate = np.zeros(len(times), np.int64)
    for event in result["events"]:
        if event.get("kind") != "official_gate_tick":
            continue
        event_time = float(event["wall_time_s"])
        target = int(event["to_active_gate"])
        gate[times >= event_time] = min(target, N_RACE_GATES - 1)
    return gate


def resample_run(run_dir: Path, args, geometry) -> dict[str, np.ndarray]:
    result = json.loads((run_dir / "result.json").read_text())
    if not bool(result["metrics"].get("official_race_finish")):
        raise ValueError(f"{run_dir.name} is not an official completed lap")

    replay = [
        json.loads(line)
        for line in (run_dir / "replay.jsonl").read_text().splitlines()
        if line.strip()
    ]
    replay_time = np.asarray([row["wall_time_s"] for row in replay], float)
    keep = np.r_[True, np.diff(replay_time) > 1e-6]
    replay_time = replay_time[keep]
    replay = [row for row, selected in zip(replay, keep, strict=True)
              if selected]

    authority_rows = [
        index for index, row in enumerate(replay)
        if bool(row.get("authority_granted"))
    ]
    if not authority_rows:
        raise ValueError(f"{run_dir.name} never granted motor authority")
    start = replay_time[authority_rows[0]]
    finish_events = [
        float(event["wall_time_s"])
        for event in result["events"]
        if event.get("kind") == "official_race_finish"
    ]
    if not finish_events:
        raise ValueError(f"{run_dir.name} has no finish event")
    finish = min(finish_events[-1], replay_time[-1])
    dt = 1.0 / float(args.hz)
    sample_time = np.arange(start, finish + 1e-9, dt)
    if finish - sample_time[-1] > 0.25 * dt:
        sample_time = np.r_[sample_time, finish]

    position_raw = np.asarray([
        row["estimate_position_ned_m"] for row in replay
    ], float)
    velocity_raw = np.asarray([
        row["estimate_velocity_ned_mps"] for row in replay
    ], float)
    sigma_raw = np.asarray([
        max(row["estimate_sigma_ned_m"]) for row in replay
    ], float)
    action_raw = np.asarray([
        wire_command_to_action(
            np.r_[row["wire_rate"], row["collective"]]
        )
        for row in replay
    ])
    rotation_raw = Rotation.from_matrix(np.asarray([
        row["attitude_body_to_ned"] for row in replay
    ], float))
    slerp = Slerp(replay_time, rotation_raw)

    def interpolate(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return np.interp(sample_time, replay_time, values)
        return np.column_stack([
            np.interp(sample_time, replay_time, values[:, axis])
            for axis in range(values.shape[1])
        ])

    position = interpolate(position_raw)
    velocity = interpolate(velocity_raw)
    sigma = interpolate(sigma_raw)
    action = np.clip(interpolate(action_raw), -1.0, 1.0).astype(np.float32)
    rotations = slerp(sample_time)
    quat_xyzw = rotations.as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    imu = [
        json.loads(line)
        for line in (run_dir / "imu_raw.jsonl").read_text().splitlines()
        if line.strip()
    ]
    imu_time = np.asarray([row["wall_time_s"] for row in imu], float)
    gyro_raw = np.asarray([row["gyro_radps"] for row in imu], float)
    gyro = np.column_stack([
        np.interp(sample_time, imu_time, gyro_raw[:, axis])
        for axis in range(3)
    ])

    gate = completed_gate_timeline(result, sample_time)
    previous_action = np.vstack([
        np.zeros(4, np.float32), action[:-1]
    ])
    observation = np.asarray([
        build_observation(
            position_world=position[index],
            quat_wxyz=quat_wxyz[index],
            velocity_world=velocity[index],
            gyro_raw=gyro[index],
            previous_action=previous_action[index],
            gate_index=int(gate[index]),
            geometry=geometry,
            position_sigma_m=float(sigma[index]),
        )
        for index in range(len(sample_time))
    ], np.float32)
    next_observation = np.vstack([
        observation[1:], observation[-1:]
    ]).astype(np.float32)
    done = np.zeros(len(sample_time), np.float32)
    done[-1] = 1.0
    elapsed = np.r_[0.0, np.diff(sample_time)]
    reward = reward_rows(
        position=position,
        gate_index=gate,
        action=action,
        elapsed_s=elapsed,
        geometry=geometry,
        progress_scale=args.progress_scale,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        time_penalty_per_s=args.time_penalty_per_s,
        action_smoothness=args.action_smoothness,
    )
    return {
        "observation": observation,
        "action": action,
        "reward": reward,
        "next_observation": next_observation,
        "done": done,
        "wall": sample_time,
        "gate_index": gate,
        "position": position,
        "velocity": velocity,
        "sigma": sigma,
        "source_episode": np.full(
            len(sample_time), run_dir.name, dtype=f"<U{len(run_dir.name)}"
        ),
    }


def recompute_base_rewards(base: dict, args, geometry) -> np.ndarray:
    wall = np.asarray(base["wall"], float)
    elapsed = np.r_[0.0, np.maximum(np.diff(wall), 0.0)]
    return reward_rows(
        position=np.asarray(base["position"], float),
        gate_index=np.asarray(base["gate_index"], np.int64),
        action=np.asarray(base["action"], np.float32),
        elapsed_s=elapsed,
        geometry=geometry,
        progress_scale=args.progress_scale,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        time_penalty_per_s=args.time_penalty_per_s,
        action_smoothness=args.action_smoothness,
    )


def apply_n_step_targets(chunk: dict[str, np.ndarray], args) -> None:
    immediate_reward = np.asarray(chunk["reward"], np.float32)
    immediate_done = np.asarray(chunk["done"], np.float32)
    immediate_next = np.asarray(chunk["next_observation"], np.float32)
    count = len(immediate_reward)
    reward = np.zeros(count, np.float32)
    done = np.zeros(count, np.float32)
    discount = np.ones(count, np.float32)
    next_observation = np.empty_like(immediate_next)
    n_step = max(1, int(args.n_step))
    gamma = float(args.gamma)
    for start in range(count):
        total = 0.0
        steps = 0
        last = start
        for offset in range(n_step):
            index = min(start + offset, count - 1)
            total += gamma ** offset * float(immediate_reward[index])
            steps += 1
            last = index
            if immediate_done[index] > 0.5:
                done[start] = 1.0
                break
        reward[start] = total
        discount[start] = gamma ** steps
        next_observation[start] = immediate_next[last]
    chunk["reward"] = reward
    chunk["done"] = done
    chunk["discount"] = discount
    chunk["next_observation"] = next_observation


def main() -> None:
    args = parser().parse_args()
    map_payload = json.loads(args.map.read_text())
    geometry = load_gate_geometry(map_payload["gates"])
    base_npz = np.load(args.base_demo, allow_pickle=False)
    source_keys = (
        "observation", "action", "reward", "next_observation", "done",
        "wall", "gate_index", "position", "velocity", "sigma",
    )
    base = {key: np.asarray(base_npz[key]) for key in source_keys}
    base["reward"] = recompute_base_rewards(base, args, geometry)
    base["source_episode"] = np.full(
        len(base["reward"]), "local_40s_clean_demo", dtype="<U20"
    )
    base_return = float(base["reward"].sum())
    apply_n_step_targets(base, args)

    converted = []
    for run_dir in sorted(args.runs_root.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "result.json").exists():
            continue
        run = resample_run(run_dir, args, geometry)
        immediate_return = float(run["reward"].sum())
        apply_n_step_targets(run, args)
        converted.append(run)
        print(
            f"{run_dir.name}: {len(converted[-1]['reward'])} rows, "
            f"return={immediate_return:.1f}",
            flush=True,
        )
    if not converted:
        raise RuntimeError("no completed GipsyDanger runs were converted")

    chunks = [base, *converted]
    keys = (*source_keys, "discount")
    output = {
        key: np.concatenate([chunk[key] for chunk in chunks])
        for key in (*keys, "source_episode")
    }
    if output["observation"].shape[1] != 53:
        raise RuntimeError(
            f"unexpected observation shape {output['observation'].shape}"
        )
    if not all(np.all(np.isfinite(output[key])) for key in keys):
        raise RuntimeError("converted demonstration contains non-finite values")
    terminal = output["done"] > 0.5
    terminal_count = int(np.count_nonzero(
        terminal & ~np.r_[False, terminal[:-1]]
    ))
    expected_terminals = 1 + len(converted)
    if terminal_count != expected_terminals:
        raise RuntimeError(
            f"expected {expected_terminals} terminal laps, got {terminal_count}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        **output,
        source_map=str(args.map),
        source_trace=str(args.runs_root),
    )
    print(
        f"wrote {args.output}: rows={len(output['reward'])}, "
        f"laps={terminal_count}, "
        f"base_return={base_return:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
