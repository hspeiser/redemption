"""Apply iterative-learning control (ILC) to a VQ2 reference trajectory.

The live trajectory tracker records the selected reference row and localized
position at every control step.  Repeated deterministic flights therefore
measure a reproducible row-wise tracking error.  This tool shifts the desired
line against that error, smooths the correction, and rebuilds the deployable
53-D observations and velocity feed-forward table.

Only rows covered by the supplied episodes are changed.  The correction is
bounded so a bad flight cannot move the reference arbitrarily far.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.rl.vq2_features import (  # noqa: E402
    MAX_RATE,
    build_observation,
    load_gate_geometry,
)


def observation_quaternion(observation: np.ndarray) -> np.ndarray:
    first = np.asarray(observation[21:24], float)
    first /= np.linalg.norm(first) + 1e-9
    second = np.asarray(observation[24:27], float)
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second) + 1e-9
    rotation = np.column_stack([first, second, np.cross(first, second)])
    return np.roll(Rotation.from_matrix(rotation).as_quat(), 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--episode", type=Path, action="append", required=True)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.65)
    parser.add_argument("--max-correction", type=float, default=0.75)
    parser.add_argument("--smooth-rows", type=float, default=7.0)
    parser.add_argument("--minimum-visits", type=int, default=1)
    parser.add_argument(
        "--gates",
        default="",
        help="Optional comma-separated target-gate segments to adapt.",
    )
    args = parser.parse_args()

    base = np.load(args.base, allow_pickle=False)
    position = np.asarray(base["position"], float)
    n_rows = len(position)
    row_errors: list[list[np.ndarray]] = [[] for _ in range(n_rows)]
    episode_summaries = []

    for episode_path in args.episode:
        episode = np.load(episode_path, allow_pickle=False)
        rows = np.asarray(episode["selected_reference_row"], int)
        actual = np.asarray(episode["position"], float)
        healthy = (
            np.asarray(episode["timing_healthy"], bool)
            if "timing_healthy" in episode.files
            else np.ones(len(rows), bool)
        )
        valid = healthy & (rows >= 0) & (rows < n_rows)
        for row in np.unique(rows[valid]):
            mask = valid & (rows == row)
            # One vote per row per episode prevents a stalled cursor from
            # dominating the aggregate merely because it repeated a row.
            row_errors[int(row)].append(
                np.median(actual[mask] - position[int(row)], axis=0)
            )
        episode_summaries.append({
            "path": str(episode_path),
            "rows": int(valid.sum()),
            "max_gate": int(np.max(episode["gate_index"][valid])),
        })

    measured = np.full((n_rows, 3), np.nan, float)
    visits = np.zeros(n_rows, int)
    for row, errors in enumerate(row_errors):
        visits[row] = len(errors)
        if len(errors) >= args.minimum_visits:
            measured[row] = np.median(np.asarray(errors), axis=0)

    known = np.flatnonzero(np.isfinite(measured[:, 0]))
    if not len(known):
        raise RuntimeError("no valid reference-row measurements found")
    # A nearest-point tracker is intentionally indifferent to along-track
    # phase.  Treating that phase lag as a geometric miss would move the line
    # forward/backward and corrupt its speed profile.  ILC should learn only
    # the cross-track component that determines gate clearance.
    base_velocity = np.asarray(base["velocity"], float)
    tangent = base_velocity / np.maximum(
        np.linalg.norm(base_velocity, axis=1, keepdims=True), 1e-6
    )
    measured[known] -= tangent[known] * np.sum(
        measured[known] * tangent[known], axis=1, keepdims=True
    )
    interpolated = np.zeros_like(measured)
    index = np.arange(n_rows)
    for axis in range(3):
        interpolated[:, axis] = np.interp(
            index, known, measured[known, axis], left=0.0, right=0.0
        )
    covered = np.zeros(n_rows, float)
    covered[known[0]:known[-1] + 1] = 1.0
    covered[: min(20, n_rows)] *= np.linspace(0.0, 1.0, min(20, n_rows))
    correction = -float(args.learning_rate) * interpolated
    correction = gaussian_filter1d(
        correction, sigma=float(args.smooth_rows), axis=0, mode="nearest"
    )
    correction *= covered[:, None]
    selected_gates = {
        int(value.strip()) for value in args.gates.split(",") if value.strip()
    }
    if selected_gates:
        gate_index = np.asarray(base["gate_index"], int)
        selected = np.isin(gate_index, sorted(selected_gates)).astype(float)
        # Ramp each selected segment in smoothly after its official gate
        # transition; do not modify the already-proven prefix.
        for gate in selected_gates:
            rows = np.flatnonzero(gate_index == gate)
            if len(rows):
                ramp_n = min(15, len(rows))
                selected[rows[:ramp_n]] *= np.linspace(0.0, 1.0, ramp_n)
        correction *= selected[:, None]
    magnitude = np.linalg.norm(correction, axis=1)
    scale = np.minimum(
        1.0,
        float(args.max_correction) / np.maximum(magnitude, 1e-9),
    )
    correction *= scale[:, None]

    adapted_position = position + correction
    wall = np.asarray(base["wall"], float)
    correction_velocity = np.gradient(correction, wall, axis=0)
    adapted_velocity = base_velocity + correction_velocity

    gates = json.loads(args.map.read_text())["gates"]
    geometry = load_gate_geometry(gates)
    old_observation = np.asarray(base["observation"], np.float32)
    gate_index = np.asarray(base["gate_index"], int)
    observation = np.zeros_like(old_observation)
    for row in range(n_rows):
        observation[row] = build_observation(
            position_world=adapted_position[row],
            quat_wxyz=observation_quaternion(old_observation[row]),
            velocity_world=adapted_velocity[row],
            gyro_raw=-old_observation[row, 27:30] * MAX_RATE,
            previous_action=old_observation[row, 30:34],
            gate_index=int(gate_index[row]),
            geometry=geometry,
            position_sigma_m=float(old_observation[row, 51] * 0.5),
        )

    payload = {key: np.asarray(base[key]) for key in base.files}
    payload.update({
        "observation": observation,
        "next_observation": np.vstack([observation[1:], observation[-1:]]),
        "position": adapted_position,
        "velocity": adapted_velocity,
        "source_episode": np.str_(
            f"ILC from {args.base.name}; {len(args.episode)} live episodes"
        ),
        "ilc_correction": correction.astype(np.float32),
        "ilc_visits": visits.astype(np.int16),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **payload)

    gate_stats = {}
    for gate in np.unique(gate_index):
        mask = gate_index == gate
        gate_stats[str(int(gate))] = {
            "rows_measured": int(np.sum(visits[mask] > 0)),
            "correction_peak_m": round(
                float(np.max(np.linalg.norm(correction[mask], axis=1))), 3
            ),
        }
    print(json.dumps({
        "out": str(args.out),
        "episodes": episode_summaries,
        "measured_rows": int(len(known)),
        "covered_rows": [int(known[0]), int(known[-1])],
        "peak_correction_m": round(float(np.max(magnitude * scale)), 3),
        "gate_stats": gate_stats,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
