"""Shared VQ2 actor observations and action scaling.

The representation deliberately contains both course-generic geometry and a
course-specific gate one-hot.  Geometry lets the policy transfer to a new
relative map; the one-hot lets it specialize its braking and turn timing for
the current VQ2 course.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation


N_RACE_GATES = 17
N_LOOKAHEAD = 3
MAX_RATE = np.deg2rad(180.0)
MAX_YAW_RATE = np.deg2rad(120.0)
RATE_GAIN = np.array([2.33, 2.35, 2.36], dtype=np.float32)
WIRE_RATE_LIMIT = np.array(
    [MAX_RATE / RATE_GAIN[0], MAX_RATE / RATE_GAIN[1],
     MAX_YAW_RATE / RATE_GAIN[2]],
    dtype=np.float32,
)
OBS_DIM = 3 * N_LOOKAHEAD + 3 * N_LOOKAHEAD + 3 + 6 + 3 + 4 + \
    N_RACE_GATES + 2
ACT_DIM = 4


@dataclass(frozen=True)
class GateGeometry:
    positions: np.ndarray
    tangents: np.ndarray


def load_gate_geometry(gates: Sequence[dict], n_gates: int = N_RACE_GATES) \
        -> GateGeometry:
    positions = np.asarray([gate["pos"] for gate in gates[:n_gates]], float)
    if positions.shape != (n_gates, 3):
        raise ValueError(
            f"expected {n_gates} gate positions, got {positions.shape}"
        )
    tangents = np.zeros_like(positions)
    for gate_index in range(n_gates):
        previous = positions[max(0, gate_index - 1)]
        following = positions[min(n_gates - 1, gate_index + 1)]
        direction = following - previous
        norm = float(np.linalg.norm(direction))
        tangents[gate_index] = (
            direction / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        )
    return GateGeometry(positions=positions, tangents=tangents)


def wire_command_to_action(command: np.ndarray) -> np.ndarray:
    """Convert logged wire body-rates/thrust to canonical [-1, 1] action."""
    command = np.asarray(command, np.float32)
    action = np.empty(4, np.float32)
    action[:3] = command[:3] / WIRE_RATE_LIMIT
    action[3] = 2.0 * command[3] - 1.0
    return np.clip(action, -1.0, 1.0)


def action_to_wire_command(action: np.ndarray) -> np.ndarray:
    """Convert canonical actor action to body-rates and collective thrust."""
    action = np.clip(np.asarray(action, np.float32), -1.0, 1.0)
    command = np.empty(4, np.float32)
    command[:3] = action[:3] * WIRE_RATE_LIMIT
    command[3] = 0.5 * (action[3] + 1.0)
    return command


def build_observation(
    *,
    position_world: np.ndarray,
    quat_wxyz: np.ndarray,
    velocity_world: np.ndarray,
    gyro_raw: np.ndarray,
    previous_action: np.ndarray,
    gate_index: int,
    geometry: GateGeometry,
    position_sigma_m: float,
) -> np.ndarray:
    """Build the 53D VQ2 SAC observation from deployable state.

    The EKF quaternion is body-to-local-course.  HIGHRES_IMU gyro is negated
    to match the corrected body convention used by the V11/EKF stack.
    """
    gate_index = int(np.clip(gate_index, 0, N_RACE_GATES - 1))
    qw, qx, qy, qz = np.asarray(quat_wxyz, float)
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    world_to_body = rotation.T
    position = np.asarray(position_world, float)

    relative = []
    tangents = []
    for lookahead in range(N_LOOKAHEAD):
        candidate = min(gate_index + lookahead, N_RACE_GATES - 1)
        relative.append(
            world_to_body @ (geometry.positions[candidate] - position)
        )
        tangents.append(world_to_body @ geometry.tangents[candidate])

    one_hot = np.zeros(N_RACE_GATES, np.float32)
    one_hot[gate_index] = 1.0
    velocity_body = world_to_body @ np.asarray(velocity_world, float)
    rates_body = -np.asarray(gyro_raw, float)
    confidence = np.array([
        np.clip(float(position_sigma_m) / 0.50, 0.0, 2.0),
        gate_index / max(N_RACE_GATES - 1, 1),
    ])
    observation = np.concatenate([
        np.concatenate(relative) / 10.0,
        np.concatenate(tangents),
        velocity_body / 10.0,
        rotation[:, 0],
        rotation[:, 1],
        rates_body / MAX_RATE,
        np.asarray(previous_action, float),
        one_hot,
        confidence,
    ]).astype(np.float32)
    if observation.shape != (OBS_DIM,):
        raise RuntimeError(
            f"VQ2 observation shape {observation.shape}, expected {(OBS_DIM,)}"
        )
    return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)


def course_progress(
    position_world: np.ndarray,
    gate_index: int,
    geometry: GateGeometry,
    spawn_position: np.ndarray,
) -> float:
    """Monotonic course coordinate for dense progress reward."""
    points = np.vstack([
        np.asarray(spawn_position, float),
        geometry.positions,
    ])
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    gate_index = int(np.clip(gate_index, 0, N_RACE_GATES - 1))
    start = points[gate_index]
    end = points[gate_index + 1]
    direction = end - start
    length = float(lengths[gate_index])
    if length <= 1e-6:
        along = 0.0
    else:
        along = float(np.dot(
            np.asarray(position_world, float) - start,
            direction / length,
        ))
    along = float(np.clip(along, 0.0, length))
    return float(lengths[:gate_index].sum() + along)

