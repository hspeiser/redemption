"""Run behavior-cloned VQ2 policy and fine-tune it with offline-between-laps SAC.

The control loop never performs gradient work while the drone is flying.
Transitions are logged immediately, failed episodes are hard-terminal, and
SAC updates run only after the simulator has been stopped.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.mavlink_io import MavIO  # noqa: E402
from aigp.flight import RATE_CMD_SIGN, RateController  # noqa: E402
from aigp.full_session_recorder import FullSessionRecorder  # noqa: E402
from aigp.rl.sac import GaussianActor, RecurrentActor, TwinCritic  # noqa: E402
from aigp.rl.vq2_env import VQ2EnvConfig, VQ2LiveEnv  # noqa: E402
from aigp.rl.vq2_features import (  # noqa: E402
    ACT_DIM,
    N_RACE_GATES,
    OBS_DIM,
    WIRE_RATE_LIMIT,
)
from aigp.rl.counterfactual_gate import (  # noqa: E402
    load_state_gated_actor,
    preserved_counterfactual_payload,
)
from aigp.vision_io import VisionRX  # noqa: E402
from aigp.vq2_live_localizer import LiveVQ2Localizer  # noqa: E402
from aigp.vq2_dashboard import VQ2Dashboard  # noqa: E402


def sha256_file(path: Path | None) -> str | None:
    """Return a stable content identity for deployed artifacts."""
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_process_cpu_affinity(mask_text: str) -> int | None:
    """Keep the trainer off CPU cores reserved for simulator physics."""
    if not mask_text or mask_text.lower() in {"all", "none"}:
        return None
    mask = int(mask_text, 0)
    if mask <= 0:
        raise ValueError("CPU affinity mask must select at least one core")
    available_mask = (1 << (os.cpu_count() or 1)) - 1
    if mask & ~available_mask:
        raise ValueError(
            f"CPU affinity {mask_text} selects unavailable cores; "
            f"this machine's mask is 0x{available_mask:X}"
        )
    if os.name == "nt":
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(
            kernel32.GetCurrentProcess(), mask
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.sched_setaffinity(
            0,
            {index for index in range(os.cpu_count() or 1) if mask >> index & 1},
        )
    return mask


def acquire_single_instance_guard():
    """Prevent two live trainers from splitting the two UDP streams."""
    if os.name != "nt":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_wchar_p,
    ]
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    handle = kernel32.CreateMutexW(
        None, False, "Local\\AIGP_VQ2_SAC_LIVE_TRAINER"
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        kernel32.CloseHandle(handle)
        raise RuntimeError(
            "another VQ2 live trainer is already running; refusing to split "
            "MAVLink and camera UDP streams between processes"
        )
    return handle


def observation_rotation(observation: np.ndarray) -> np.ndarray:
    """Recover the body-to-course rotation encoded in an observation."""
    first = np.asarray(observation[21:24], float)
    second = np.asarray(observation[24:27], float)
    first /= np.linalg.norm(first) + 1e-9
    second -= first * np.dot(first, second)
    second /= np.linalg.norm(second) + 1e-9
    third = np.cross(first, second)
    return np.column_stack([first, second, third])


def estimate_demo_acceleration(
    velocity: np.ndarray,
    wall: np.ndarray,
    source_episode: np.ndarray | None,
) -> np.ndarray:
    """Estimate smooth per-row world acceleration without crossing laps."""
    velocity = np.asarray(velocity, float)
    wall = np.asarray(wall, float)
    acceleration = np.zeros_like(velocity)
    if source_episode is None or np.asarray(source_episode).ndim == 0:
        groups = [np.arange(len(velocity))]
    else:
        source_episode = np.asarray(source_episode).astype(str)
        groups = [
            np.flatnonzero(source_episode == name)
            for name in dict.fromkeys(source_episode.tolist())
        ]
    for indices in groups:
        if len(indices) < 3:
            continue
        t = wall[indices]
        v = velocity[indices]
        dt = float(np.median(np.diff(t)))
        if not np.isfinite(dt) or dt <= 1e-4:
            dt = 1.0 / 30.0
        window = min(15, len(indices) if len(indices) % 2 else len(indices) - 1)
        if window >= 5:
            smooth_v = savgol_filter(
                v,
                window_length=window,
                polyorder=min(3, window - 2),
                axis=0,
                mode="interp",
            )
            smooth_a = savgol_filter(
                smooth_v,
                window_length=window,
                polyorder=min(3, window - 2),
                deriv=1,
                delta=dt,
                axis=0,
                mode="interp",
            )
        else:
            smooth_a = np.gradient(v, dt, axis=0)
        magnitude = np.linalg.norm(smooth_a, axis=1)
        scale = np.minimum(1.0, 18.0 / np.maximum(magnitude, 1e-6))
        acceleration[indices] = smooth_a * scale[:, None]
    return acceleration.astype(np.float32)


def load_reference_demo(path: Path) -> dict[str, np.ndarray]:
    """Load every array that defines the live nearest-row reference."""
    data = np.load(path, allow_pickle=False)
    observation = np.asarray(data["observation"], np.float32)
    velocity = np.asarray(data["velocity"], np.float32)
    position = np.asarray(data["position"], np.float32)
    gate = np.asarray(data["gate_index"], np.int16)
    rotation = np.asarray([
        observation_rotation(row) for row in observation
    ], np.float32)
    gate_vector_world = np.einsum(
        "nij,nj->ni", rotation, observation[:, :3]
    ) * 10.0
    gate_position = np.zeros((17, 3), np.float32)
    for gate_index in range(17):
        mask = gate == gate_index
        if not np.any(mask):
            raise ValueError(f"reference {path} has no gate {gate_index}")
        gate_position[gate_index] = np.median(
            position[mask] + gate_vector_world[mask], axis=0
        )
    return {
        "demo_observation": observation,
        "demo_action": np.asarray(data["action"], np.float32),
        "demo_velocity": velocity,
        "demo_acceleration": estimate_demo_acceleration(
            velocity,
            np.asarray(data["wall"], np.float64),
            (
                np.asarray(data["source_episode"])
                if "source_episode" in data.files else None
            ),
        ),
        "demo_position": position,
        "demo_gate": gate,
        "demo_rotation": rotation,
        "demo_gate_vector_world": gate_vector_world,
        "demo_gate_position": gate_position,
    }


def parse_gate_value_pairs(
    text: str,
    *,
    value_type: type = float,
) -> tuple[tuple[int, float | int], ...]:
    """Parse comma-separated ``gate:value`` controller overrides."""
    pairs = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        if ":" not in item:
            raise ValueError(
                f"expected gate:value entry, got {item!r}"
            )
        gate_text, value_text = item.split(":", 1)
        gate = int(gate_text.strip())
        if not 0 <= gate < N_RACE_GATES:
            raise ValueError(
                f"gate override {gate} is outside 0..{N_RACE_GATES - 1}"
            )
        pairs.append((gate, value_type(value_text.strip())))
    return tuple(pairs)


def parse_gate_phase_windows(
    text: str,
) -> tuple[tuple[int, float, float], ...]:
    """Parse comma-separated ``gate:start_phase:end_phase`` windows."""
    windows = []
    for item in str(text).split(","):
        if not item.strip():
            continue
        fields = item.split(":")
        if len(fields) != 3:
            raise ValueError(
                "expected gate:start_phase:end_phase entry, "
                f"got {item!r}"
            )
        gate = int(fields[0])
        start, end = float(fields[1]), float(fields[2])
        if not 0 <= gate < N_RACE_GATES:
            raise ValueError(f"residual phase gate {gate} is outside course")
        if not 0.0 <= start <= end <= 1.0:
            raise ValueError(
                f"invalid residual phase window {item!r}; require "
                "0 <= start <= end <= 1"
            )
        windows.append((gate, start, end))
    return tuple(windows)


def gate_crossing_offset(
    gate: dict,
    position_world: np.ndarray,
) -> dict[str, float]:
    """Express a crossing position in the gate's local aperture frame."""
    qw, qx, qy, qz = np.asarray(gate["quat_wxyz"], float)
    rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    local = rotation.T @ (
        np.asarray(position_world, float)
        - np.asarray(gate["pos"], float)
    )
    return {
        "lateral_m": float(local[0]),
        "plane_m": float(local[1]),
        "vertical_m": float(local[2]),
    }


def gate10_cut_metrics(transitions: list[dict]) -> dict[str, float | None]:
    """Summarize the active x=121..125 gate-10 S-curve."""
    rows = [
        row for row in transitions
        if int(row["gate_index"]) == 10
        and 121.0 <= float(row["position"][0]) <= 125.0
    ]
    if len(rows) < 2:
        return {
            "gate10_cut_slope_y_per_x": None,
            "gate10_cut_x_span_m": 0.0,
            "gate10_peak_abs_lateral_action": None,
            "gate10_peak_abs_raw_lateral_feedback": None,
            "gate10_peak_abs_clipped_lateral_feedback": None,
            "gate10_lateral_feedback_saturation_fraction": None,
        }
    positions = np.asarray([row["position"] for row in rows], float)
    left = int(np.argmin(np.abs(positions[:, 0] - 121.0)))
    right = int(np.argmin(np.abs(positions[:, 0] - 125.0)))
    dx = float(positions[right, 0] - positions[left, 0])
    slope = (
        float((positions[right, 1] - positions[left, 1]) / dx)
        if abs(dx) > 0.5 else None
    )
    raw = np.asarray([
        row["raw_lateral_feedback"] for row in rows
    ], float)
    clipped = np.asarray([
        row["clipped_lateral_feedback"] for row in rows
    ], float)
    lateral_action = np.asarray([
        row["wire_action"][0] for row in rows
    ], float)
    return {
        "gate10_cut_slope_y_per_x": slope,
        "gate10_cut_x_span_m": abs(dx),
        "gate10_peak_abs_lateral_action": float(
            np.max(np.abs(lateral_action))
        ),
        "gate10_peak_abs_raw_lateral_feedback": float(
            np.max(np.abs(raw))
        ),
        "gate10_peak_abs_clipped_lateral_feedback": float(
            np.max(np.abs(clipped))
        ),
        "gate10_lateral_feedback_saturation_fraction": float(
            np.mean(np.abs(raw) >= 0.20 - 1e-6)
        ),
    }


def gate_control_metrics(transitions: list[dict]) -> dict[str, dict]:
    """Expose per-gate feedback margin, including gates 11, 13, and 15."""
    metrics = {}
    for gate_index in range(N_RACE_GATES):
        rows = [
            row for row in transitions
            if int(row["gate_index"]) == gate_index
        ]
        if not rows:
            continue
        raw = np.asarray([
            row["raw_lateral_feedback"] for row in rows
        ], float)
        clipped = np.asarray([
            row["clipped_lateral_feedback"] for row in rows
        ], float)
        position_error = np.asarray([
            row["lateral_position_error_m"] for row in rows
        ], float)
        lateral_action = np.asarray([
            row["wire_action"][0] for row in rows
        ], float)
        metrics[str(gate_index)] = {
            "steps": len(rows),
            "gain_scale": float(rows[-1][
                "effective_lateral_gain_scale"
            ]),
            "action_lead": int(rows[-1][
                "effective_reference_action_lead"
            ]),
            "peak_abs_lateral_action": float(
                np.max(np.abs(lateral_action))
            ),
            "peak_abs_raw_feedback": float(np.max(np.abs(raw))),
            "peak_abs_clipped_feedback": float(
                np.max(np.abs(clipped))
            ),
            "feedback_saturation_fraction": float(
                np.mean(np.abs(raw) >= 0.20 - 1e-6)
            ),
            "peak_abs_position_error_m": float(
                np.max(np.abs(position_error))
            ),
        }
    return metrics


class ReplayMemory:
    def __init__(self, capacity: int, observation_dim: int) -> None:
        self.capacity = int(capacity)
        self.observation = np.zeros(
            (capacity, observation_dim), np.float32
        )
        self.action = np.zeros((capacity, ACT_DIM), np.float32)
        self.reward = np.zeros(capacity, np.float32)
        self.next_observation = np.zeros(
            (capacity, observation_dim), np.float32
        )
        self.done = np.zeros(capacity, np.float32)
        self.discount = np.ones(capacity, np.float32)
        self.gate = np.zeros(capacity, np.int16)
        self.is_demo = np.zeros(capacity, np.float32)
        self.priority = np.ones(capacity, np.float32)
        self.size = 0
        self.cursor = 0

    def add(
        self,
        observation,
        action,
        reward,
        next_observation,
        done,
        gate,
        discount=1.0,
        priority=1.0,
        is_demo=False,
    ) -> None:
        index = self.cursor
        self.observation[index] = observation
        self.action[index] = action
        self.reward[index] = reward
        self.next_observation[index] = next_observation
        self.done[index] = done
        self.discount[index] = discount
        self.gate[index] = gate
        self.is_demo[index] = float(is_demo)
        self.priority[index] = max(float(priority), 1e-3)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_arrays(self, payload: dict[str, np.ndarray]) -> None:
        for row in range(len(payload["reward"])):
            self.add(
                payload["observation"][row],
                payload["action"][row],
                payload["reward"][row],
                payload["next_observation"][row],
                payload["done"][row],
                payload["gate_index"][row],
                discount=(
                    payload["discount"][row]
                    if "discount" in payload else 1.0
                ),
                priority=1.0 + abs(float(payload["reward"][row])),
                is_demo=(
                    bool(payload["is_demo"][row])
                    if "is_demo" in payload else True
                ),
            )

    def save(self, path: Path) -> None:
        """Atomically persist the populated replay rows across live restarts."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            temporary,
            observation=self.observation[:self.size],
            action=self.action[:self.size],
            reward=self.reward[:self.size],
            next_observation=self.next_observation[:self.size],
            done=self.done[:self.size],
            discount=self.discount[:self.size],
            gate_index=self.gate[:self.size],
            is_demo=self.is_demo[:self.size],
            priority=self.priority[:self.size],
        )
        os.replace(temporary, path)

    def load(self, path: Path, *, default_discount: float) -> int:
        archive = np.load(path, allow_pickle=False)
        # Materialize each compressed member once. Indexing NpzFile by key
        # inside the row loop would decompress the complete member for every
        # transition and turn a 13 MB restart into a multi-minute stall.
        payload = {key: np.asarray(archive[key]) for key in archive.files}
        archive.close()
        count = len(payload["reward"])
        for row in range(count):
            self.add(
                payload["observation"][row],
                payload["action"][row],
                payload["reward"][row],
                payload["next_observation"][row],
                payload["done"][row],
                payload["gate_index"][row],
                discount=(
                    payload["discount"][row]
                    if "discount" in payload else default_discount
                ),
                priority=(
                    payload["priority"][row]
                    if "priority" in payload else 1.0
                ),
                is_demo=(
                    bool(payload["is_demo"][row])
                    if "is_demo" in payload else False
                ),
            )
        return count

    def sample(
        self,
        count: int,
        *,
        prioritized: bool,
        gate_balanced: bool = False,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self.size == 0:
            raise RuntimeError("cannot sample an empty replay")
        weights = np.ones(self.size, np.float64)
        if prioritized:
            weights *= np.power(self.priority[:self.size], 0.60)
        if gate_balanced:
            counts = np.bincount(
                self.gate[:self.size].astype(int), minlength=17
            )
            weights *= 1.0 / np.maximum(
                counts[self.gate[:self.size].astype(int)], 1
            )
        weights /= weights.sum()
        indices = np.random.choice(
            self.size, size=count, replace=True, p=weights
        )
        return indices, {
            "observation": self.observation[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_observation": self.next_observation[indices],
            "done": self.done[indices],
            "discount": self.discount[indices],
            "gate_index": self.gate[indices],
            "is_demo": self.is_demo[indices],
        }


class VQ2SACLearner:
    def __init__(
        self,
        checkpoint_path: Path,
        demo_path: Path,
        *,
        line_path: Path | None,
        line_map_path: Path,
        line_model_path: Path,
        line_speed_cap: float,
        line_clearance: float,
        device: str,
        actor_lr: float,
        warmup_actor_lr: float,
        critic_lr: float,
        alpha: float,
        bc_weight: float,
        sac_actor_weight: float,
        trust_weight: float,
        teacher_blend: float,
        teacher_blends: tuple[tuple[int, float], ...],
        actor_speed_limits: tuple[tuple[int, float], ...],
        actor_speed_governor_margin: float,
        residual_scale: float,
        residual_scale_gates: tuple[tuple[int, float], ...],
        residual_output_clip: float,
        macro_exploration_gates: tuple[int, ...],
        macro_lateral_residual_scale: float,
        zero_actor_output: bool,
        actor_warmup_updates: int,
        longitudinal_position_gain: float,
        longitudinal_velocity_gain: float,
        longitudinal_position_gains: tuple[tuple[int, float], ...],
        longitudinal_velocity_gains: tuple[tuple[int, float], ...],
        lateral_position_gain: float,
        lateral_velocity_gain: float,
        special_lateral_gate: int,
        special_lateral_gain_scale: float,
        lateral_gain_scales: tuple[tuple[int, float], ...],
        lateral_feedback_limits: tuple[tuple[int, float], ...],
        lateral_bias_gates: tuple[int, ...],
        lateral_action_bias: float,
        right_lateral_bias_gates: tuple[int, ...],
        right_lateral_action_bias: float,
        extra_lateral_biases: tuple[tuple[int, float], ...],
        vertical_position_gain: float,
        vertical_velocity_gain: float,
        special_vertical_gate: int,
        special_vertical_gain_scale: float,
        vertical_bias_gates: tuple[int, ...],
        vertical_action_bias: float,
        extra_vertical_biases: tuple[tuple[int, float], ...],
        trajectory_blend: float,
        trajectory_blend_gates: tuple[int, ...],
        trajectory_blends: tuple[tuple[int, float], ...],
        trajectory_kp_scale: float,
        trajectory_kv_scale: float,
        trajectory_attitude_gain: float,
        reference_mode: str,
        reference_feedback_scale: float,
        reference_thrust_scale: float,
        reference_thrust_scales: tuple[tuple[int, float], ...],
        reference_rate_scale: float,
        reference_rate_scales: tuple[tuple[int, float], ...],
        reference_velocity_scale: float,
        reference_velocity_scales: tuple[tuple[int, float], ...],
        reference_lateral_offsets: tuple[tuple[int, float], ...],
        reference_vertical_offsets: tuple[tuple[int, float], ...],
        reference_sequential_speed: float,
        reference_sequential_speeds: tuple[tuple[int, float], ...],
        predictive_handoff_distances: tuple[tuple[int, float], ...],
        gate_center_funnel_gates: tuple[int, ...],
        gate_center_funnel_distance: float,
        gate_center_funnel_full_distance: float,
        gate_center_funnel_strength: float,
        reference_action_lead: int,
        gate4_action_lead: int,
        reference_action_leads: tuple[tuple[int, int], ...],
        reference_max_advance: int,
        reference_max_retreat: int,
        train_gate: int,
        residual_gates: tuple[int, ...],
        residual_phase_windows: tuple[tuple[int, float, float], ...],
        frozen_residual_episode: Path | None,
        frozen_residual_gates: tuple[int, ...],
        frozen_action_episode: Path | None,
        frozen_action_gates: tuple[int, ...],
        actor_objective: str,
        awr_weight: float,
        awr_temperature: float,
        awr_max_weight: float,
        positive_td_priority: float,
        finish_replay_boost: float,
        n_step: int,
        ppo_residual_checkpoint: Path | None,
        secondary_ppo_residual_checkpoint: Path | None,
        secondary_ppo_residual_gates: tuple[int, ...],
        ppo_residual_schedule: Path | None,
        champion_demo_path: Path | None,
        champion_config_path: Path | None,
    ) -> None:
        self.device = torch.device(device)
        payload = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        is_residual = payload.get("kind") == "vq2_residual_sac"

        # The demonstrated full-action policy is frozen as part of the
        # baseline controller. SAC learns only a bounded correction around
        # that baseline, in exactly the action coordinates seen by its critic.
        teacher_state = (
            payload["teacher_actor"] if is_residual else payload["actor"]
        )
        self.teacher = GaussianActor(OBS_DIM, ACT_DIM).cpu().eval()
        self.teacher.load_state_dict(teacher_state)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.recurrent_teacher = None
        self.recurrent_teacher_hidden = None
        if "recurrent_teacher_actor" in payload:
            recurrent_metadata = payload.get("recurrent_teacher", {})
            self.recurrent_teacher = RecurrentActor(
                OBS_DIM,
                ACT_DIM,
                hidden_dim=int(recurrent_metadata.get("hidden_dim", 128)),
            ).cpu().eval()
            self.recurrent_teacher.load_state_dict(
                payload["recurrent_teacher_actor"]
            )
            for parameter in self.recurrent_teacher.parameters():
                parameter.requires_grad_(False)
        self.recurrent_residual = None
        self.recurrent_residual_hidden = None
        if "recurrent_residual_actor" in payload:
            recurrent_metadata = payload.get("recurrent_residual", {})
            self.recurrent_residual = RecurrentActor(
                OBS_DIM,
                ACT_DIM,
                hidden_dim=int(recurrent_metadata.get("hidden_dim", 128)),
            ).cpu().eval()
            self.recurrent_residual.load_state_dict(
                payload["recurrent_residual_actor"]
            )
            for parameter in self.recurrent_residual.parameters():
                parameter.requires_grad_(False)

        self.actor = GaussianActor(OBS_DIM, ACT_DIM).to(self.device)
        if is_residual:
            self.actor.load_state_dict(payload["actor"])
            if zero_actor_output:
                # A curriculum checkpoint may only have deployed its actor on
                # one gate.  Its unconstrained outputs at all other gates are
                # therefore not a safe full-course initialization.  Preserve
                # the learned feature backbone and critic, but make the new
                # residual policy exactly reproduce the reference controller
                # until fresh full-course evidence moves the output head.
                with torch.no_grad():
                    self.actor.mean.weight.zero_()
                    self.actor.mean.bias.zero_()
        else:
            # GaussianActor initializes to a zero-mean residual. Give it
            # useful but tightly bounded initial stochastic exploration.
            with torch.no_grad():
                self.actor.log_std.bias.fill_(-2.0)
        # Keep live action inference on CPU.  The MLP takes sub-millisecond
        # time there and cannot be starved by the simulator/GateNet sharing
        # the GPU near a gate.
        self.inference_actor = copy.deepcopy(self.actor).cpu().eval()
        self.counterfactual_inference_actor = load_state_gated_actor(
            payload,
            self.inference_actor,
            observation_dim=OBS_DIM,
            actor_factory=lambda: GaussianActor(OBS_DIM, ACT_DIM),
            device="cpu",
        )
        self.counterfactual_checkpoint_payload = (
            preserved_counterfactual_payload(payload)
        )
        self.last_counterfactual_gate_score = 0.0
        self.last_counterfactual_gate_active = False
        self.critic = TwinCritic(OBS_DIM, ACT_DIM).to(self.device)
        if is_residual:
            self.critic.load_state_dict(payload["critic"])
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        if is_residual:
            self.target_critic.load_state_dict(payload["target_critic"])
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)
        self.observation_mean = np.asarray(
            payload["observation_mean"], np.float32
        )
        self.observation_std = np.asarray(
            payload["observation_std"], np.float32
        )
        self.normalization_clip: float | None = None
        self.ppo_residual_schedule_path = ppo_residual_schedule
        self.ppo_residual_schedule: np.ndarray | None = None
        self.secondary_ppo_residual_actor = None
        self.secondary_ppo_observation_mean: np.ndarray | None = None
        self.secondary_ppo_observation_std: np.ndarray | None = None
        self.secondary_ppo_residual_gates = {
            int(gate)
            for gate in secondary_ppo_residual_gates
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.probe_arm = "candidate"
        if ppo_residual_checkpoint is not None:
            ppo = torch.load(
                ppo_residual_checkpoint, map_location=self.device,
                weights_only=False,
            )
            self.actor.load_state_dict(ppo["actor"])
            self.inference_actor.load_state_dict(ppo["actor"])
            self.observation_mean = np.asarray(
                ppo["obs_mean"].detach().cpu(), np.float32
            )
            self.observation_std = np.sqrt(
                np.asarray(ppo["obs_var"].detach().cpu(), np.float32)
                + 1e-6
            )
            self.normalization_clip = 8.0
            print(
                "Loaded fastsim PPO residual "
                f"{ppo_residual_checkpoint} @ iter {ppo.get('iter', '?')}",
                flush=True,
            )
        if secondary_ppo_residual_checkpoint is not None:
            secondary_ppo = torch.load(
                secondary_ppo_residual_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            secondary_actor = GaussianActor(OBS_DIM, ACT_DIM).cpu().eval()
            secondary_actor.load_state_dict(secondary_ppo["actor"])
            for parameter in secondary_actor.parameters():
                parameter.requires_grad_(False)
            self.secondary_ppo_residual_actor = secondary_actor
            self.secondary_ppo_observation_mean = np.asarray(
                secondary_ppo["obs_mean"].detach().cpu(), np.float32
            )
            self.secondary_ppo_observation_std = np.sqrt(
                np.asarray(
                    secondary_ppo["obs_var"].detach().cpu(), np.float32
                )
                + 1e-6
            )
            print(
                "Loaded gate-routed secondary fastsim PPO residual "
                f"{secondary_ppo_residual_checkpoint} @ iter "
                f"{secondary_ppo.get('iter', '?')} for gates "
                f"{sorted(self.secondary_ppo_residual_gates)}",
                flush=True,
            )
        self.return_scale = float(payload["return_scale"])
        self.gamma = float(payload["gamma"])
        self.alpha = float(alpha)
        self.bc_weight = float(bc_weight)
        self.sac_actor_weight = float(sac_actor_weight)
        self.trust_weight = float(trust_weight)
        self.teacher_blend = float(teacher_blend)
        self.teacher_blends = {
            int(gate): float(np.clip(blend, 0.0, 1.0))
            for gate, blend in teacher_blends
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.actor_speed_limits = {
            int(gate): max(float(limit), 0.0)
            for gate, limit in actor_speed_limits
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.actor_speed_governor_margin = max(
            float(actor_speed_governor_margin), 1e-3
        )
        self.residual_scale = float(residual_scale)
        self.residual_scale_gates = {
            int(gate): max(float(scale), 0.0)
            for gate, scale in residual_scale_gates
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.residual_output_clip = float(np.clip(
            residual_output_clip,
            0.0,
            1.0,
        ))
        self.macro_exploration_gates = tuple(
            int(gate) for gate in macro_exploration_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        self.macro_lateral_residual_scale = max(
            float(macro_lateral_residual_scale), 0.0
        )
        self.actor_warmup_updates = int(actor_warmup_updates)
        self.longitudinal_position_gain = max(
            float(longitudinal_position_gain), 0.0
        )
        self.longitudinal_velocity_gain = max(
            float(longitudinal_velocity_gain), 0.0
        )
        self.longitudinal_position_gains = {
            int(gate): max(float(gain), 0.0)
            for gate, gain in longitudinal_position_gains
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.longitudinal_velocity_gains = {
            int(gate): max(float(gain), 0.0)
            for gate, gain in longitudinal_velocity_gains
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.actor_lr = float(actor_lr)
        self.warmup_actor_lr = float(warmup_actor_lr)
        self.lateral_position_gain = float(lateral_position_gain)
        self.lateral_velocity_gain = float(lateral_velocity_gain)
        self.special_lateral_gate = int(special_lateral_gate)
        self.special_lateral_gain_scale = max(
            float(special_lateral_gain_scale), 0.0
        )
        self.lateral_gain_scales = {
            int(gate): max(float(scale), 0.0)
            for gate, scale in lateral_gain_scales
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.lateral_feedback_limits = {
            int(gate): float(np.clip(limit, 0.0, 1.0))
            for gate, limit in lateral_feedback_limits
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.lateral_bias_gates = frozenset(
            int(gate) for gate in lateral_bias_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        self.lateral_action_bias = float(lateral_action_bias)
        self.right_lateral_bias_gates = frozenset(
            int(gate) for gate in right_lateral_bias_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        self.right_lateral_action_bias = float(
            right_lateral_action_bias
        )
        self.extra_lateral_biases = {
            int(gate): float(bias)
            for gate, bias in extra_lateral_biases
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.vertical_position_gain = float(vertical_position_gain)
        self.vertical_velocity_gain = float(vertical_velocity_gain)
        self.special_vertical_gate = int(special_vertical_gate)
        self.special_vertical_gain_scale = max(
            float(special_vertical_gain_scale), 0.0
        )
        self.vertical_bias_gates = frozenset(
            int(gate) for gate in vertical_bias_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        self.vertical_action_bias = float(vertical_action_bias)
        self.extra_vertical_biases = {
            int(gate): float(bias)
            for gate, bias in extra_vertical_biases
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.trajectory_blend = float(np.clip(trajectory_blend, 0.0, 1.0))
        self.trajectory_blend_gates = frozenset(
            int(gate) for gate in trajectory_blend_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        self.trajectory_blends = {
            int(gate): float(np.clip(blend, 0.0, 1.0))
            for gate, blend in trajectory_blends
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.trajectory_controller = RateController(thrust_limit=0.52)
        self.trajectory_kp_scale = max(float(trajectory_kp_scale), 0.0)
        self.trajectory_kv_scale = max(float(trajectory_kv_scale), 0.0)
        self.trajectory_attitude_gain = max(
            float(trajectory_attitude_gain), 0.0
        )
        self.trajectory_controller.kp *= self.trajectory_kp_scale
        self.trajectory_controller.kv *= self.trajectory_kv_scale
        self.trajectory_controller.k_att = self.trajectory_attitude_gain
        # Convert desired physical body rates through the measured simulator
        # inner-loop gain.  A single historical 2.56x scalar materially
        # over-commanded VQ2 yaw (fresh full-record sysid measures ~2.00x).
        # The model is read even when the optional FlatRefController is off,
        # because trajectory_blend uses this conversion independently.
        from aigp.fastsim.sysid import SurrogateModel

        trajectory_model = SurrogateModel.load(str(line_model_path))
        self.trajectory_rate_cmd_gain = 1.0 / np.maximum(
            np.abs(np.asarray(trajectory_model.rate_gain, np.float32)),
            1e-3,
        )
        self.reference_mode = str(reference_mode)
        self.reference_feedback_scale = max(
            float(reference_feedback_scale), 0.0
        )
        self.reference_thrust_scale = max(
            float(reference_thrust_scale), 0.0
        )
        self.reference_thrust_scales = {
            int(gate): max(float(scale), 0.0)
            for gate, scale in reference_thrust_scales
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.reference_rate_scale = max(float(reference_rate_scale), 0.0)
        self.reference_rate_scales = {
            int(gate): max(float(scale), 0.0)
            for gate, scale in reference_rate_scales
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.reference_velocity_scale = max(
            float(reference_velocity_scale), 0.0
        )
        self.reference_velocity_scales = {
            int(gate): max(float(scale), 0.0)
            for gate, scale in reference_velocity_scales
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.reference_lateral_offsets = {
            int(gate): float(offset)
            for gate, offset in reference_lateral_offsets
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.reference_vertical_offsets = {
            int(gate): float(offset)
            for gate, offset in reference_vertical_offsets
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.reference_sequential_speed = max(
            float(reference_sequential_speed), 0.05
        )
        self.reference_sequential_speeds = {
            int(gate): max(float(speed), 0.05)
            for gate, speed in reference_sequential_speeds
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.predictive_handoff_distances = {
            int(gate): max(float(distance), 0.0)
            for gate, distance in predictive_handoff_distances
            if 0 <= int(gate) < N_RACE_GATES - 1
        }
        self.gate_center_funnel_gates = {
            int(gate) for gate in gate_center_funnel_gates
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.gate_center_funnel_distance = max(
            float(gate_center_funnel_distance), 0.0
        )
        self.gate_center_funnel_full_distance = max(
            float(gate_center_funnel_full_distance), 0.0
        )
        self.gate_center_funnel_strength = float(np.clip(
            gate_center_funnel_strength, 0.0, 1.0
        ))
        self.reference_action_lead = int(reference_action_lead)
        self.gate4_action_lead = int(gate4_action_lead)
        self.reference_action_leads = {
            int(gate): int(lead)
            for gate, lead in reference_action_leads
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.reference_max_advance = max(1, int(reference_max_advance))
        self.reference_max_retreat = max(0, int(reference_max_retreat))
        self.train_gate = int(train_gate)
        self.residual_gates = frozenset(
            int(gate) for gate in residual_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        self.residual_phase_windows = {
            int(gate): (float(start), float(end))
            for gate, start, end in residual_phase_windows
            if 0 <= int(gate) < N_RACE_GATES
        }
        self.frozen_residual_profiles: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        frozen_gate_set = frozenset(
            int(gate) for gate in frozen_residual_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        if frozen_residual_episode is not None:
            frozen_episode = np.load(
                frozen_residual_episode, allow_pickle=False
            )
            frozen_gate = np.asarray(
                frozen_episode["gate_index"], np.int16
            )
            frozen_row = np.asarray(
                frozen_episode["reference_row"], np.int32
            )
            frozen_action = np.asarray(
                frozen_episode["action"], np.float32
            )
            for gate in sorted(frozen_gate_set):
                selected = frozen_gate == gate
                if not np.any(selected):
                    raise ValueError(
                        f"frozen residual episode has no gate {gate} rows"
                    )
                rows = frozen_row[selected]
                actions = frozen_action[selected]
                unique_rows = np.unique(rows)
                profile = np.stack([
                    actions[rows == row].mean(axis=0)
                    for row in unique_rows
                ]).astype(np.float32)
                self.frozen_residual_profiles[gate] = (
                    unique_rows.astype(np.float32),
                    profile,
                )
        self.frozen_action_profiles: dict[
            int, tuple[np.ndarray, np.ndarray]
        ] = {}
        frozen_action_gate_set = frozenset(
            int(gate) for gate in frozen_action_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
        if frozen_action_episode is not None:
            frozen_episode = np.load(
                frozen_action_episode, allow_pickle=False
            )
            frozen_gate = np.asarray(
                frozen_episode["gate_index"], np.int16
            )
            frozen_row = np.asarray(
                frozen_episode["reference_row"], np.int32
            )
            frozen_action = np.asarray(
                frozen_episode["wire_action"], np.float32
            )
            for gate in sorted(frozen_action_gate_set):
                selected = frozen_gate == gate
                if not np.any(selected):
                    raise ValueError(
                        f"frozen action episode has no gate {gate} rows"
                    )
                rows = frozen_row[selected]
                actions = frozen_action[selected]
                unique_rows = np.unique(rows)
                profile = np.stack([
                    actions[rows == row].mean(axis=0)
                    for row in unique_rows
                ]).astype(np.float32)
                self.frozen_action_profiles[gate] = (
                    unique_rows.astype(np.float32),
                    profile,
                )
        self.actor_objective = str(actor_objective)
        self.awr_weight = float(awr_weight)
        self.awr_temperature = max(float(awr_temperature), 1e-6)
        self.awr_max_weight = max(float(awr_max_weight), 1.0)
        self.positive_td_priority = max(float(positive_td_priority), 0.0)
        self.finish_replay_boost = max(
            float(finish_replay_boost), 1.0
        )
        self.n_step = max(1, int(n_step))
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=1e-6
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=1e-6
        )
        self.demo = ReplayMemory(100_000, OBS_DIM)
        demo_npz = np.load(demo_path, allow_pickle=False)
        candidate_reference = load_reference_demo(Path(demo_path))
        self.reference_demo_tables = {"candidate": candidate_reference}
        if champion_demo_path is not None:
            self.reference_demo_tables["protected_champion"] = (
                load_reference_demo(Path(champion_demo_path))
            )
        for name, value in candidate_reference.items():
            setattr(self, name, value)
        gate_map = json.loads(Path(line_map_path).read_text())["gates"]
        self.schedule_gate_position = np.asarray([
            gate["pos"] for gate in gate_map[:17]
        ], np.float32)
        self.schedule_track_points = np.vstack([
            np.asarray([0.0, 0.0, -0.3], np.float32),
            self.schedule_gate_position,
        ])
        if ppo_residual_schedule is not None:
            schedule_payload = json.loads(
                Path(ppo_residual_schedule).read_text()
            )
            schedule = np.asarray(
                schedule_payload["residual_schedule"], np.float32
            )
            if (
                schedule.ndim != 3
                or schedule.shape[1] < 2
                or schedule.shape[2] != ACT_DIM
                or schedule.shape[0] > len(self.schedule_gate_position)
            ):
                raise ValueError(
                    "PPO residual schedule must have shape "
                    f"[gates, knots>=2, {ACT_DIM}], got {schedule.shape}"
                )
            self.ppo_residual_schedule = schedule
            print(
                "Loaded fastsim residual schedule "
                f"{ppo_residual_schedule} with shape {schedule.shape}",
                flush=True,
            )
        gate_quaternion = np.asarray([
            gate["quat_wxyz"] for gate in gate_map[:17]
        ], np.float64)
        self.gate_rotation = Rotation.from_quat(np.stack([
            gate_quaternion[:, 1], gate_quaternion[:, 2],
            gate_quaternion[:, 3], gate_quaternion[:, 0],
        ], axis=1)).as_matrix()
        # Match FastVQ2Env/lineopt's course-oriented aperture frames exactly.
        # Local x is lateral and local z is vertical. Keep this separate from
        # gate_rotation because the latter is used by existing diagnostics.
        reference_gate_rotation = self.gate_rotation.copy()
        for gate_index in range(N_RACE_GATES):
            incoming = self.schedule_gate_position[gate_index] - (
                self.schedule_gate_position[gate_index - 1]
                if gate_index else np.zeros(3, np.float32)
            )
            if np.dot(
                reference_gate_rotation[gate_index, :, 1], incoming
            ) < 0.0:
                reference_gate_rotation[gate_index, :, 0] *= -1.0
                reference_gate_rotation[gate_index, :, 1] *= -1.0
        self.reference_gate_offsets_world = np.zeros(
            (N_RACE_GATES, 3), np.float32
        )
        for gate_index in range(N_RACE_GATES):
            self.reference_gate_offsets_world[gate_index] = (
                self.reference_lateral_offsets.get(gate_index, 0.0)
                * reference_gate_rotation[gate_index, :, 0]
                + self.reference_vertical_offsets.get(gate_index, 0.0)
                * reference_gate_rotation[gate_index, :, 2]
            )
        # A protected live arm must switch the *whole* controller, not just
        # zero the residual actor.  Keep immutable profiles in this learner so
        # A/B episodes share one process, one localizer, and one simulator
        # health trajectory without silently retaining candidate gains/biases.
        self.controller_profiles = {
            "candidate": self._capture_controller_profile()
        }
        if champion_config_path is not None:
            self.controller_profiles["protected_champion"] = (
                self._controller_profile_from_config(
                    Path(champion_config_path), reference_gate_rotation
                )
            )
        self.line_backbone = None
        self.line_path = line_path
        if line_path is not None:
            from aigp.fastsim.lineopt import (
                FlatRefController,
                LineConfig,
                N_GATES as LINEOPT_GATES,
                build_reference,
                feedforward_actions,
                load_oriented_gates,
            )
            from aigp.fastsim.sysid import SurrogateModel

            best = np.load(line_path, allow_pickle=False)
            theta = np.asarray(best["theta"], np.float64)
            gate_position, gate_rotation = load_oriented_gates(
                line_map_path
            )
            line_config = LineConfig(
                speed_cap=float(line_speed_cap),
                clearance=float(line_clearance),
            )
            line_reference = build_reference(
                gate_position,
                gate_rotation,
                theta[:LINEOPT_GATES * 2],
                theta[LINEOPT_GATES * 2:],
                line_config,
            )
            line_model = SurrogateModel.load(str(line_model_path))
            line_feedforward = feedforward_actions(
                line_reference, line_model
            )
            # This is the exact geometric tracker used by the live lineopt
            # validation harness. SAC learns only a bounded residual around
            # it; the line's closed-loop actions must never be replayed
            # through the legacy demo tracker (that double-counts feedback).
            self.line_backbone = FlatRefController(
                line_reference,
                line_feedforward,
                n_envs=1,
                device="cpu",
                speed_cap=float(line_speed_cap),
                model=line_model,
                trim0=-0.045,
                lead=6,
            )
            print(
                f"LINE SAC backbone: {line_path.name}, "
                f"{self.line_backbone.n_pts} rows, planned lap "
                f"{line_reference['planned_lap_s']:.1f}s, "
                f"cap {float(line_speed_cap):.1f}",
                flush=True,
            )
        self.teacher_features = np.r_[
            np.arange(0, 9),
            np.arange(18, 34),
        ]
        self.demo_teacher_state = self.normalize(
            self.demo_observation
        )[:, self.teacher_features]
        self.demo.add_arrays({
            "observation": np.asarray(demo_npz["observation"]),
            # Demonstrations define the baseline, so their ideal residual is
            # exactly zero.
            "action": np.zeros_like(demo_npz["action"], dtype=np.float32),
            "reward": np.asarray(demo_npz["reward"]),
            "next_observation": np.asarray(demo_npz["next_observation"]),
            "done": np.asarray(demo_npz["done"]),
            "gate_index": np.asarray(demo_npz["gate_index"]),
            "discount": (
                np.asarray(demo_npz["discount"], np.float32)
                if "discount" in demo_npz.files
                else np.full(
                    len(demo_npz["reward"]), self.gamma, np.float32
                )
            ),
        })
        self.live = ReplayMemory(500_000, OBS_DIM)
        self.exploration_state = np.zeros(ACT_DIM, np.float32)
        self.episode_macro_gate: int | None = None
        self.episode_macro_lateral_residual = 0.0
        self.reference_gate: int | None = None
        self.reference_cursor: int | None = None
        self.predictive_handoff_latched_gate: int | None = None
        self.last_control_debug: dict[str, float | int | bool] = {}
        self.last_reference_action = np.zeros(ACT_DIM, np.float32)
        self.updates = int(payload.get("updates", 0)) if is_residual else 0
        if is_residual and not zero_actor_output:
            if "actor_optimizer" in payload:
                self.actor_optimizer.load_state_dict(
                    payload["actor_optimizer"]
                )
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(
                    payload["critic_optimizer"]
                )
        # Checkpoints preserve optimizer moments, but a new curriculum run
        # must be able to deliberately lower its learning rates.
        for group in self.actor_optimizer.param_groups:
            group["lr"] = float(actor_lr)
        for group in self.critic_optimizer.param_groups:
            group["lr"] = float(critic_lr)

    def normalize(self, observation: np.ndarray) -> np.ndarray:
        normalized = (
            np.asarray(observation, np.float32) - self.observation_mean
        ) / self.observation_std
        if self.normalization_clip is not None:
            normalized = np.clip(
                normalized, -self.normalization_clip,
                self.normalization_clip,
            )
        return normalized

    def begin_episode(self, *, explore: bool = False) -> None:
        self.exploration_state.fill(0.0)
        self.episode_macro_gate = None
        self.episode_macro_lateral_residual = 0.0
        if (
            explore
            and self.macro_exploration_gates
            and self.macro_lateral_residual_scale > 0.0
        ):
            self.episode_macro_gate = int(np.random.choice(
                self.macro_exploration_gates
            ))
            sign = -1.0 if np.random.random() < 0.5 else 1.0
            magnitude = (
                0.5 if np.random.random() < 0.25 else 1.0
            )
            self.episode_macro_lateral_residual = float(
                sign * magnitude * self.macro_lateral_residual_scale
            )
        self.reference_gate = None
        self.reference_cursor = None
        self.predictive_handoff_latched_gate = None
        self.last_control_debug = {}
        self.last_reference_action.fill(0.0)
        self.recurrent_teacher_hidden = None
        self.recurrent_residual_hidden = None
        self.trajectory_controller.zi = 0.0
        if self.line_backbone is not None:
            # action() runs under torch.inference_mode(), so the tracker's
            # cursor becomes an inference tensor after its first update.
            # Reset it in the same mode between episodes.
            with torch.inference_mode():
                self.line_backbone.reset(torch.tensor([0]))

    def set_probe_arm(self, arm: str) -> None:
        """Select the protected control arm without reloading the process."""
        if arm not in {"candidate", "protected_champion"}:
            raise ValueError(f"unknown probe arm: {arm}")
        self.probe_arm = arm
        table = self.reference_demo_tables.get(
            arm, self.reference_demo_tables["candidate"]
        )
        for name, value in table.items():
            setattr(self, name, value)
        profile = self.controller_profiles.get(arm)
        if arm == "protected_champion" and profile is None:
            raise RuntimeError(
                "protected champion arm has no exact controller profile; "
                "pass --interleave-champion-config"
            )
        if profile is not None:
            self._apply_controller_profile(profile)

    _CONTROLLER_PROFILE_FIELDS = (
        "teacher_blend",
        "teacher_blends",
        "actor_speed_limits",
        "actor_speed_governor_margin",
        "longitudinal_position_gain",
        "longitudinal_velocity_gain",
        "longitudinal_position_gains",
        "longitudinal_velocity_gains",
        "lateral_position_gain",
        "lateral_velocity_gain",
        "special_lateral_gate",
        "special_lateral_gain_scale",
        "lateral_gain_scales",
        "lateral_feedback_limits",
        "lateral_bias_gates",
        "lateral_action_bias",
        "right_lateral_bias_gates",
        "right_lateral_action_bias",
        "extra_lateral_biases",
        "vertical_position_gain",
        "vertical_velocity_gain",
        "special_vertical_gate",
        "special_vertical_gain_scale",
        "vertical_bias_gates",
        "vertical_action_bias",
        "extra_vertical_biases",
        "trajectory_blend",
        "trajectory_blend_gates",
        "trajectory_blends",
        "trajectory_kp_scale",
        "trajectory_kv_scale",
        "trajectory_attitude_gain",
        "reference_mode",
        "reference_feedback_scale",
        "reference_thrust_scale",
        "reference_thrust_scales",
        "reference_rate_scale",
        "reference_rate_scales",
        "reference_velocity_scale",
        "reference_velocity_scales",
        "reference_lateral_offsets",
        "reference_vertical_offsets",
        "reference_sequential_speed",
        "reference_sequential_speeds",
        "predictive_handoff_distances",
        "gate_center_funnel_gates",
        "gate_center_funnel_distance",
        "gate_center_funnel_full_distance",
        "gate_center_funnel_strength",
        "reference_action_lead",
        "gate4_action_lead",
        "reference_action_leads",
        "reference_max_advance",
        "reference_max_retreat",
        "reference_gate_offsets_world",
    )

    def _capture_controller_profile(self) -> dict:
        profile = {
            name: copy.deepcopy(getattr(self, name))
            for name in self._CONTROLLER_PROFILE_FIELDS
        }
        profile["trajectory_controller_kp"] = np.asarray(
            self.trajectory_controller.kp
        ).copy()
        profile["trajectory_controller_kv"] = np.asarray(
            self.trajectory_controller.kv
        ).copy()
        profile["trajectory_controller_k_att"] = float(
            self.trajectory_controller.k_att
        )
        return profile

    def _apply_controller_profile(self, profile: dict) -> None:
        for name in self._CONTROLLER_PROFILE_FIELDS:
            setattr(self, name, copy.deepcopy(profile[name]))
        self.trajectory_controller.kp = np.asarray(
            profile["trajectory_controller_kp"]
        ).copy()
        self.trajectory_controller.kv = np.asarray(
            profile["trajectory_controller_kv"]
        ).copy()
        self.trajectory_controller.k_att = float(
            profile["trajectory_controller_k_att"]
        )

    def _controller_profile_from_config(
        self,
        path: Path,
        reference_gate_rotation: np.ndarray,
    ) -> dict:
        payload = json.loads(path.read_text())
        cfg = payload.get("args", payload)

        def pairs(name: str, value_type=float) -> dict:
            return {
                int(gate): value_type(value)
                for gate, value in parse_gate_value_pairs(
                    str(cfg.get(name, "")), value_type=value_type
                )
                if 0 <= int(gate) < N_RACE_GATES
            }

        def gates(name: str) -> frozenset[int]:
            return frozenset(
                int(value.strip())
                for value in str(cfg.get(name, "")).split(",")
                if value.strip() and 0 <= int(value.strip()) < N_RACE_GATES
            )

        profile = self._capture_controller_profile()
        scalar_fields = {
            "teacher_blend": (float, 1.0),
            "actor_speed_governor_margin": (float, 1.0),
            "longitudinal_position_gain": (float, 0.0),
            "longitudinal_velocity_gain": (float, 0.0),
            "lateral_position_gain": (float, 0.08),
            "lateral_velocity_gain": (float, 0.04),
            "special_lateral_gate": (int, -1),
            "special_lateral_gain_scale": (float, 1.0),
            "lateral_action_bias": (float, 0.0),
            "right_lateral_action_bias": (float, 0.0),
            "vertical_position_gain": (float, 0.30),
            "vertical_velocity_gain": (float, 0.10),
            "special_vertical_gate": (int, -1),
            "special_vertical_gain_scale": (float, 1.0),
            "vertical_action_bias": (float, 0.0),
            "trajectory_blend": (float, 0.0),
            "trajectory_kp_scale": (float, 1.0),
            "trajectory_kv_scale": (float, 1.0),
            "trajectory_attitude_gain": (float, 4.0),
            "reference_feedback_scale": (float, 1.0),
            "reference_thrust_scale": (float, 1.0),
            "reference_rate_scale": (float, 1.0),
            "reference_velocity_scale": (float, 1.0),
            "reference_sequential_speed": (float, 1.0),
            "gate_center_funnel_distance": (float, 0.0),
            "gate_center_funnel_full_distance": (float, 1.5),
            "gate_center_funnel_strength": (float, 1.0),
            "reference_action_lead": (int, 0),
            "gate4_action_lead": (int, 0),
            "reference_max_advance": (int, 8),
            "reference_max_retreat": (int, 2),
        }
        for name, (cast, default) in scalar_fields.items():
            profile[name] = cast(cfg.get(name, default))
        profile["reference_mode"] = str(
            cfg.get("reference_mode", "nearest")
        )
        pair_fields = {
            "teacher_blends": float,
            "actor_speed_limits": float,
            "longitudinal_position_gains": float,
            "longitudinal_velocity_gains": float,
            "lateral_gain_scales": float,
            "lateral_feedback_limits": float,
            "extra_lateral_biases": float,
            "extra_vertical_biases": float,
            "trajectory_blends": float,
            "reference_thrust_scales": float,
            "reference_rate_scales": float,
            "reference_velocity_scales": float,
            "reference_lateral_offsets": float,
            "reference_vertical_offsets": float,
            "reference_sequential_speeds": float,
            "predictive_handoff_distances": float,
            "reference_action_leads": int,
        }
        for name, cast in pair_fields.items():
            profile[name] = pairs(name, cast)
        for name in (
            "lateral_bias_gates",
            "right_lateral_bias_gates",
            "vertical_bias_gates",
            "trajectory_blend_gates",
            "gate_center_funnel_gates",
        ):
            profile[name] = gates(name)
        offsets_world = np.zeros((N_RACE_GATES, 3), np.float32)
        for gate_index in range(N_RACE_GATES):
            offsets_world[gate_index] = (
                profile["reference_lateral_offsets"].get(gate_index, 0.0)
                * reference_gate_rotation[gate_index, :, 0]
                + profile["reference_vertical_offsets"].get(gate_index, 0.0)
                * reference_gate_rotation[gate_index, :, 2]
            )
        profile["reference_gate_offsets_world"] = offsets_world
        controller = RateController(thrust_limit=0.52)
        controller.kp *= max(profile["trajectory_kp_scale"], 0.0)
        controller.kv *= max(profile["trajectory_kv_scale"], 0.0)
        controller.k_att = max(profile["trajectory_attitude_gain"], 0.0)
        profile["trajectory_controller_kp"] = np.asarray(
            controller.kp
        ).copy()
        profile["trajectory_controller_kv"] = np.asarray(
            controller.kv
        ).copy()
        profile["trajectory_controller_k_att"] = float(controller.k_att)
        return profile

    def _residual_gate_active(self, gate_index: int) -> bool:
        if self.residual_gates:
            return gate_index in self.residual_gates
        return self.train_gate < 0 or gate_index == self.train_gate

    def _frozen_residual(
        self, gate_index: int, reference_row: int
    ) -> np.ndarray | None:
        profile = self.frozen_residual_profiles.get(gate_index)
        if profile is None:
            return None
        rows, actions = profile
        return np.asarray([
            np.interp(float(reference_row), rows, actions[:, axis])
            for axis in range(ACT_DIM)
        ], np.float32)

    def _frozen_action(
        self, gate_index: int, reference_row: int
    ) -> np.ndarray | None:
        profile = self.frozen_action_profiles.get(gate_index)
        if profile is None:
            return None
        rows, actions = profile
        return np.asarray([
            np.interp(float(reference_row), rows, actions[:, axis])
            for axis in range(ACT_DIM)
        ], np.float32)

    def _scheduled_residual_correction(
        self,
        gate_index: int,
        current_position_world: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Match fastsim's gate-segment phase interpolation exactly."""
        table = self.ppo_residual_schedule
        if (
            self.probe_arm == "protected_champion"
            or table is None
            or not 0 <= gate_index < len(table)
        ):
            return np.zeros(ACT_DIM, np.float32), 0.0
        start = self.schedule_track_points[gate_index]
        end = self.schedule_track_points[gate_index + 1]
        segment = end - start
        length_sq = float(np.dot(segment, segment))
        phase = float(np.clip(
            np.dot(current_position_world - start, segment)
            / max(length_sq, 1e-9),
            0.0,
            1.0,
        ))
        coordinate = phase * (table.shape[1] - 1)
        left = int(np.floor(coordinate))
        right = min(left + 1, table.shape[1] - 1)
        fraction = np.float32(coordinate - left)
        correction = (
            table[gate_index, left]
            + fraction
            * (table[gate_index, right] - table[gate_index, left])
        )
        return correction.astype(np.float32, copy=False), phase

    @torch.inference_mode()
    def action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        exploration_clip: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        event_gate_index = int(np.argmax(observation[34:51]))
        tensor = torch.from_numpy(
            self.normalize(observation)
        ).unsqueeze(0)
        secondary_actor_active = bool(
            self.secondary_ppo_residual_actor is not None
            and event_gate_index in self.secondary_ppo_residual_gates
        )
        if secondary_actor_active:
            secondary_normalized = np.clip(
                (
                    np.asarray(observation, np.float32)
                    - self.secondary_ppo_observation_mean
                ) / self.secondary_ppo_observation_std,
                -8.0,
                8.0,
            ).astype(np.float32, copy=False)
            residual_mean = self.secondary_ppo_residual_actor.deterministic(
                torch.from_numpy(secondary_normalized).unsqueeze(0)
            )
            self.last_counterfactual_gate_score = 0.0
            self.last_counterfactual_gate_active = False
        elif self.recurrent_residual is not None:
            residual_mean, self.recurrent_residual_hidden = (
                self.recurrent_residual.step(
                    tensor, self.recurrent_residual_hidden
                )
            )
            self.last_counterfactual_gate_score = 0.0
            self.last_counterfactual_gate_active = False
        elif self.counterfactual_inference_actor is None:
            residual_mean = self.inference_actor.deterministic(tensor)
            self.last_counterfactual_gate_score = 0.0
            self.last_counterfactual_gate_active = False
        else:
            gate_score = self.counterfactual_inference_actor.gate_score(tensor)
            residual_mean = self.counterfactual_inference_actor.deterministic(
                tensor
            )
            self.last_counterfactual_gate_score = float(gate_score.item())
            self.last_counterfactual_gate_active = bool(
                gate_score.item()
                >= self.counterfactual_inference_actor.threshold
            )
        if self.recurrent_teacher is not None:
            teacher_mean, self.recurrent_teacher_hidden = (
                self.recurrent_teacher.step(
                    tensor, self.recurrent_teacher_hidden
                )
            )
        else:
            teacher_mean = self.teacher.deterministic(tensor)
        current_rotation = observation_rotation(observation)
        current_gate_vector_world = (
            current_rotation @ np.asarray(observation[:3]) * 10.0
        )
        gate_index = event_gate_index
        event_position_world = (
            self.schedule_gate_position[event_gate_index]
            - current_gate_vector_world
        )
        handoff_distance = self.predictive_handoff_distances.get(
            event_gate_index, 0.0
        )
        if (
            self.predictive_handoff_latched_gate is not None
            and self.predictive_handoff_latched_gate != event_gate_index
        ):
            # The authoritative gate event caught up with the geometric
            # handoff.  Release the latch so the new gate can arm its own
            # handoff independently.
            self.predictive_handoff_latched_gate = None
        predictive_handoff_triggered = (
            handoff_distance > 0.0
            and event_gate_index < N_RACE_GATES - 1
            and np.linalg.norm(current_gate_vector_world) <= handoff_distance
        )
        if predictive_handoff_triggered:
            # Gate events can arrive several control frames after the drone
            # has crossed the plane.  Once we switch the controller to the
            # next gate, never fall back to the stale gate merely because the
            # distance starts increasing on the far side of the plane.
            self.predictive_handoff_latched_gate = event_gate_index
        predictive_handoff_active = (
            self.predictive_handoff_latched_gate == event_gate_index
        )
        if predictive_handoff_active:
            current_position_world = (
                self.demo_gate_position[event_gate_index]
                - current_gate_vector_world
            )
            gate_index = event_gate_index + 1
            current_gate_vector_world = (
                self.demo_gate_position[gate_index]
                - current_position_world
            )
        gate_candidates = np.flatnonzero(self.demo_gate == gate_index)
        if not len(gate_candidates):
            raise RuntimeError(
                f"clean demonstration has no samples for gate {gate_index}"
            )
        gate_start = int(gate_candidates[0])
        gate_end = int(gate_candidates[-1])
        segment_rows = max(gate_end - gate_start, 1)
        current_reference_offset = self.reference_gate_offsets_world[
            gate_index
        ]
        previous_reference_offset = (
            self.reference_gate_offsets_world[gate_index - 1]
            if gate_index > 0 else np.zeros(3, np.float32)
        )

        def geometry_offsets(rows: np.ndarray) -> np.ndarray:
            phase = (
                (np.asarray(rows, np.float32) - float(gate_start))
                / float(segment_rows)
            )
            return (
                previous_reference_offset[None, :]
                + phase[:, None]
                * (
                    current_reference_offset
                    - previous_reference_offset
                )[None, :]
            )

        sequential_speed = self.reference_sequential_speeds.get(
            gate_index, self.reference_sequential_speed
        )
        candidates = gate_candidates
        if self.reference_gate != gate_index:
            self.reference_gate = gate_index
            self.reference_cursor = int(candidates[0])
        cursor = float(self.reference_cursor)
        sequential_reference_active = (
            self.reference_mode == "sequential"
            or gate_index in self.reference_sequential_speeds
        )
        if sequential_reference_active:
            lower = int(np.clip(np.floor(cursor), gate_start, gate_end))
            upper = int(np.clip(np.ceil(cursor), gate_start, gate_end))
            if lower == upper:
                nearest = np.asarray([lower], dtype=int)
                weights = np.ones(1, dtype=float)
            else:
                fraction = float(cursor - np.floor(cursor))
                nearest = np.asarray([lower, upper], dtype=int)
                weights = np.asarray([1.0 - fraction, fraction], float)
            self.reference_cursor = min(
                cursor + sequential_speed, float(gate_end)
            )
        else:
            cursor = int(cursor)
            admissible = candidates[
                (candidates >= cursor - self.reference_max_retreat)
                & (candidates <= cursor + self.reference_max_advance)
            ]
            if not len(admissible):
                # Gate sections are contiguous in the clean demonstration.
                admissible = candidates[candidates >= cursor]
            if not len(admissible):
                admissible = candidates[-1:]
            candidates = admissible
            candidate_gate_vectors = (
                self.demo_gate_vector_world[candidates]
                - geometry_offsets(candidates)
            )
            gate_vector_error = (
                candidate_gate_vectors
                - current_gate_vector_world[None, :]
            )
            previous_action_error = np.sum(
                (
                    self.demo_observation[candidates, 30:34]
                    - observation[30:34]
                ) ** 2,
                axis=1,
            )
            speed_error = (
                np.linalg.norm(
                    self.demo_observation[candidates, 18:21],
                    axis=1,
                )
                - np.linalg.norm(observation[18:21])
            ) ** 2
            distance = (
                4.0 * np.sum(gate_vector_error**2, axis=1)
                + 2.0 * previous_action_error
                + 0.5 * speed_error
            )
            best_local = int(np.argmin(distance))
            self.reference_cursor = int(candidates[best_local])
            nearest_local = np.argsort(distance)[:4]
            nearest = candidates[nearest_local]
            weights = 1.0 / np.maximum(
                distance[nearest_local], 1e-4
            )
            weights /= weights.sum()
        legacy_action_lead = self.reference_action_lead
        if gate_index == 4:
            legacy_action_lead += self.gate4_action_lead
        action_lead = self.reference_action_leads.get(
            gate_index, legacy_action_lead
        )
        action_rows = np.clip(
            nearest + action_lead,
            gate_start,
            gate_end,
        )
        reference = torch.from_numpy(np.sum(
            self.demo_action[action_rows] * weights[:, None],
            axis=0,
        )).unsqueeze(0)
        # A time-compressed trajectory requires more than advancing through
        # the demo rows faster. Desired body rates scale linearly with time
        # compression; acceleration (handled by thrust scaling below) scales
        # approximately quadratically. Keep this feed-forward-only so the
        # live tracking correction retains its independently tuned authority.
        rate_scale = self.reference_rate_scales.get(
            gate_index, self.reference_rate_scale
        )
        if rate_scale != 1.0:
            reference[:, :3] = torch.clamp(
                rate_scale * reference[:, :3], -1.0, 1.0
            )
        reference_observation = np.sum(
            self.demo_observation[nearest] * weights[:, None],
            axis=0,
        )
        # Feedback around the demonstrated line in one consistent frame.
        # Gate-relative vectors from two observations live in two different
        # body frames, so subtracting them directly flips signs in turns.
        reference_rotation = observation_rotation(reference_observation)
        reference_gate_vector_world = (
            reference_rotation
            @ np.asarray(reference_observation[:3])
            * 10.0
        )
        reference_geometry_offset_world = np.sum(
            geometry_offsets(nearest) * weights[:, None], axis=0
        )
        reference_gate_vector_world -= reference_geometry_offset_world
        current_position_world = (
            self.demo_gate_position[gate_index]
            - current_gate_vector_world
        )
        reference_position_world = (
            self.demo_gate_position[gate_index]
            - reference_gate_vector_world
        )
        gate_normal = self.gate_rotation[gate_index, :, 1]
        plane_distance = abs(float(np.dot(
            current_position_world - self.demo_gate_position[gate_index],
            gate_normal,
        )))
        funnel_weight = 0.0
        if (
            self.gate_center_funnel_distance > 0.0
            and (
                not self.gate_center_funnel_gates
                or gate_index in self.gate_center_funnel_gates
            )
            and plane_distance <= self.gate_center_funnel_distance
        ):
            start = self.gate_center_funnel_distance
            full = min(self.gate_center_funnel_full_distance, start - 1e-3)
            funnel_weight = self.gate_center_funnel_strength * float(
                np.clip((start - plane_distance) / max(start - full, 1e-3),
                        0.0, 1.0)
            )
            reference_from_center = (
                reference_position_world
                - self.demo_gate_position[gate_index]
            )
            in_plane_reference = (
                reference_from_center
                - gate_normal * np.dot(reference_from_center, gate_normal)
            )
            reference_position_world = (
                reference_position_world
                - funnel_weight * in_plane_reference
            )
        # Preserve the controller's established error convention:
        # current minus reference.  Gate-relative vectors point from drone to
        # gate, so their historical subtraction is the same quantity.
        position_error_world = (
            current_position_world - reference_position_world
        )
        current_velocity_world = (
            current_rotation
            @ np.asarray(observation[18:21])
            * 10.0
        )
        reference_velocity_world = np.sum(
            self.demo_velocity[nearest] * weights[:, None],
            axis=0,
        )
        reference_acceleration_world = np.sum(
            self.demo_acceleration[nearest] * weights[:, None],
            axis=0,
        )
        velocity_scale = self.reference_velocity_scales.get(
            gate_index, self.reference_velocity_scale
        )
        trajectory_velocity_world = (
            velocity_scale * reference_velocity_world
        )
        trajectory_acceleration_world = (
            velocity_scale * velocity_scale
            * reference_acceleration_world
        )
        velocity_error_world = (
            current_velocity_world - reference_velocity_world
        )
        attitude_error = Rotation.from_matrix(
            current_rotation.T @ reference_rotation
        ).as_rotvec()
        position_error_reference = (
            reference_rotation.T @ position_error_world
        )
        velocity_error_reference = (
            reference_rotation.T @ velocity_error_world
        )
        sequential_velocity_error_reference = (
            reference_rotation.T
            @ (
                current_velocity_world
                - sequential_speed * reference_velocity_world
            )
        )
        tracking = np.zeros(ACT_DIM, np.float32)
        tracking[:3] = np.clip(
            np.asarray(attitude_error)
            * np.array([0.70, 0.70, 0.60]),
            -0.25,
            0.25,
        )
        longitudinal_position_gain = self.longitudinal_position_gains.get(
            gate_index, self.longitudinal_position_gain
        )
        longitudinal_velocity_gain = self.longitudinal_velocity_gains.get(
            gate_index, self.longitudinal_velocity_gain
        )
        raw_longitudinal_feedback = (
            longitudinal_position_gain
            * position_error_reference[0]
            + longitudinal_velocity_gain
            * sequential_velocity_error_reference[0]
        )
        clipped_longitudinal_feedback = float(np.clip(
            raw_longitudinal_feedback, -0.20, 0.20
        ))
        tracking[1] += clipped_longitudinal_feedback
        legacy_lateral_gain_scale = (
            self.special_lateral_gain_scale
            if gate_index == self.special_lateral_gate
            else 1.0
        )
        lateral_gain_scale = self.lateral_gain_scales.get(
            gate_index, legacy_lateral_gain_scale
        )
        raw_lateral_feedback = (
            -lateral_gain_scale
            * self.lateral_position_gain
            * position_error_reference[1]
            - lateral_gain_scale
            * self.lateral_velocity_gain
            * velocity_error_reference[1]
        )
        lateral_feedback_limit = self.lateral_feedback_limits.get(
            gate_index, 0.20
        )
        clipped_lateral_feedback = float(np.clip(
            raw_lateral_feedback,
            -lateral_feedback_limit,
            lateral_feedback_limit,
        ))
        tracking[0] += clipped_lateral_feedback
        if gate_index in self.lateral_bias_gates:
            tracking[0] = np.clip(
                tracking[0] + self.lateral_action_bias,
                -0.25,
                0.25,
            )
        if gate_index in self.right_lateral_bias_gates:
            tracking[0] = np.clip(
                tracking[0] + self.right_lateral_action_bias,
                -0.25,
                0.25,
            )
        if gate_index in self.extra_lateral_biases:
            tracking[0] = np.clip(
                tracking[0] + self.extra_lateral_biases[gate_index],
                -0.25,
                0.25,
            )
        vertical_gain_scale = (
            self.special_vertical_gain_scale
            if gate_index == self.special_vertical_gate
            else 1.0
        )
        vertical_position_feedback = (
            vertical_gain_scale
            * self.vertical_position_gain
            * position_error_world[2]
        )
        tracking[3] = np.clip(
            vertical_position_feedback
            + vertical_gain_scale
            * self.vertical_velocity_gain
            * velocity_error_world[2],
            -0.15,
            0.15,
        )
        if gate_index in self.vertical_bias_gates:
            # Positive canonical thrust lifts the NED-frame trajectory. Keep
            # this separate from the map/localizer so a course-line correction
            # cannot distort landmark geometry.
            tracking[3] = np.clip(
                tracking[3] + self.vertical_action_bias,
                -0.15,
                0.15,
            )
        if gate_index in self.extra_vertical_biases:
            tracking[3] = np.clip(
                tracking[3] + self.extra_vertical_biases[gate_index],
                -0.15,
                0.15,
            )
        reference = reference + self.reference_feedback_scale * (
            torch.from_numpy(tracking).unsqueeze(0)
        )
        thrust_scale = self.reference_thrust_scales.get(
            gate_index, self.reference_thrust_scale
        )
        if thrust_scale != 1.0:
            # Scale only thrust above/below hover rather than multiplying
            # collective outright. This slows the racing line without
            # starving the vehicle of gravity compensation.
            reference_thrust = 0.5 * (reference[:, 3] + 1.0)
            reference_thrust = (
                0.25
                + thrust_scale
                * (reference_thrust - 0.25)
            )
            reference[:, 3] = torch.clamp(
                2.0 * reference_thrust - 1.0,
                -1.0,
                1.0,
            )
        effective_trajectory_blend = self.trajectory_blends.get(
            gate_index, self.trajectory_blend
        )
        trajectory_blend_active = (
            effective_trajectory_blend > 0.0
            and (
                not self.trajectory_blend_gates
                or gate_index in self.trajectory_blend_gates
            )
        )
        if trajectory_blend_active:
            yaw_des = float(np.arctan2(
                reference_rotation[1, 0],
                reference_rotation[0, 0],
            ))
            desired_rates, desired_thrust, _ = (
                self.trajectory_controller.update(
                    current_position_world,
                    current_velocity_world,
                    current_rotation,
                    reference_position_world,
                    trajectory_velocity_world,
                    yaw_des,
                    1.0 / 30.0,
                    a_ref=trajectory_acceleration_world,
                )
            )
            trajectory_action = np.empty(ACT_DIM, np.float32)
            trajectory_action[:3] = (
                RATE_CMD_SIGN
                * self.trajectory_rate_cmd_gain
                * desired_rates
            ) / WIRE_RATE_LIMIT
            trajectory_action[3] = 2.0 * desired_thrust - 1.0
            trajectory_action = np.clip(
                trajectory_action, -1.0, 1.0
            )
            reference = (
                (1.0 - effective_trajectory_blend) * reference
                + effective_trajectory_blend
                * torch.from_numpy(trajectory_action).unsqueeze(0)
            )
        # Preserve the protected controller's answer for the *same live
        # observation*, even when a student/direct actor is selected below.
        # This is the shadow-teacher label required by on-policy DAgger.
        self.last_reference_action = reference.squeeze(0).detach().cpu(
        ).numpy().astype(np.float32, copy=True)
        line_backbone_active = self.line_backbone is not None
        if line_backbone_active:
            # Official race status, represented by the observation one-hot,
            # is authoritative. This prevents a noisy plane-crossing belief
            # from moving the line cursor to the wrong segment at a reversal.
            self.line_backbone.cur_gate[0] = gate_index
            current_position_world = (
                self.demo_gate_position[gate_index]
                - current_gate_vector_world
            )
            line_action = self.line_backbone.action(
                torch.from_numpy(current_position_world).float()[None],
                torch.from_numpy(current_velocity_world).float()[None],
                torch.from_numpy(current_rotation).float()[None],
            )
            base = line_action
            self.reference_cursor = int(self.line_backbone.idx[0])
        else:
            effective_teacher_blend = self.teacher_blends.get(
                gate_index, self.teacher_blend
            )
            actor_speed_limit = self.actor_speed_limits.get(gate_index)
            actor_speed_governor = 0.0
            if actor_speed_limit is not None:
                current_speed = float(np.linalg.norm(
                    np.asarray(observation[18:21], float) * 10.0
                ))
                actor_speed_governor = float(np.clip(
                    (
                        current_speed
                        - (
                            actor_speed_limit
                            - self.actor_speed_governor_margin
                        )
                    ) / self.actor_speed_governor_margin,
                    0.0,
                    1.0,
                ))
                effective_teacher_blend = max(
                    effective_teacher_blend, actor_speed_governor
                )
            base = (
                effective_teacher_blend * reference
                + (1.0 - effective_teacher_blend) * teacher_mean
            )
        schedule_correction, schedule_phase = (
            self._scheduled_residual_correction(
                event_gate_index, event_position_world
            )
        )
        phase_window = self.residual_phase_windows.get(gate_index)
        residual_active = self._residual_gate_active(gate_index) and (
            phase_window is None
            or phase_window[0] <= schedule_phase <= phase_window[1]
        )
        frozen_residual = self._frozen_residual(
            gate_index, int(self.reference_cursor)
        )
        if not residual_active:
            residual = torch.zeros_like(residual_mean)
        elif frozen_residual is not None and gate_index != self.train_gate:
            # Reuse the exact bounded correction from a successful solved
            # segment while exploration and learning move to a later gate.
            residual = torch.from_numpy(
                frozen_residual
            ).unsqueeze(0)
        elif deterministic or (
            self.train_gate >= 0 and gate_index != self.train_gate
        ):
            # Previously solved curriculum gates may keep their learned mean,
            # but exploration belongs only on the gate currently being
            # trained. Otherwise promoting gate 15 re-randomizes gate 5 and
            # makes deep-course data collection needlessly rare.
            residual = torch.clamp(
                residual_mean,
                -self.residual_output_clip,
                self.residual_output_clip,
            )
        else:
            # SAC is off-policy: collection noise need not be the actor's
            # frame-wise Gaussian. Smooth bounded noise preserves a racing
            # line while still probing nearby residual actions.
            clip = max(float(exploration_clip), 0.0)
            rho = 0.96
            innovation = (
                clip * np.sqrt(1.0 - rho * rho)
                * np.random.randn(ACT_DIM)
            )
            self.exploration_state = np.clip(
                rho * self.exploration_state + innovation,
                -clip,
                clip,
            ).astype(np.float32)
            residual = torch.clamp(
                residual_mean
                + torch.from_numpy(self.exploration_state).unsqueeze(0),
                -self.residual_output_clip,
                self.residual_output_clip,
            )
        macro_lateral_residual = (
            self.episode_macro_lateral_residual
            if gate_index == self.episode_macro_gate
            else 0.0
        )
        if macro_lateral_residual:
            macro = torch.zeros_like(residual)
            macro[:, 0] = float(macro_lateral_residual)
            # Store the macro perturbation in the residual action itself.
            # Passed-gate SIL/AWR can therefore imitate a coherent successful
            # line change instead of seeing an unexplained baseline offset.
            residual = torch.clamp(
                residual + macro,
                -self.residual_output_clip,
                self.residual_output_clip,
            )
        if (
            self.ppo_residual_schedule is not None
            and self.probe_arm == "candidate"
        ):
            # Fastsim searches in normalized residual coordinates, then the
            # environment applies the configured residual scale. Preserve
            # that order here so an offline-screened schedule is not silently
            # attenuated by the legacy live residual-output clip.
            residual = torch.clamp(
                residual_mean
                + torch.from_numpy(schedule_correction).unsqueeze(0),
                -1.0,
                1.0,
            )
        if self.probe_arm == "protected_champion":
            # Same-session control arm: reproduce the known protected
            # reference controller exactly. The candidate actor and schedule
            # stay loaded, avoiding a process/session-health confound.
            residual = torch.zeros_like(residual_mean)
        frozen_action = (
            None
            if self.probe_arm == "protected_champion"
            else self._frozen_action(gate_index, int(self.reference_cursor))
        )
        if frozen_action is not None and gate_index != self.train_gate:
            selected = torch.from_numpy(
                np.clip(frozen_action, -1.0, 1.0)
            ).unsqueeze(0)
        else:
            effective_residual_scale = (
                self.residual_scale
                * self.residual_scale_gates.get(gate_index, 1.0)
            )
            selected = torch.clamp(
                base + effective_residual_scale * residual,
                -1.0,
                1.0,
            )
        self.last_control_debug = {
            "event_gate_index": int(event_gate_index),
            "control_gate_index": int(gate_index),
            "predictive_handoff_active": bool(
                predictive_handoff_active
            ),
            "predictive_handoff_triggered": bool(
                predictive_handoff_triggered
            ),
            "gate_center_funnel_weight": float(funnel_weight),
            "gate_plane_distance_m": float(plane_distance),
            "effective_lateral_gain_scale": float(lateral_gain_scale),
            "effective_reference_action_lead": int(action_lead),
            "effective_reference_velocity_scale": float(velocity_scale),
            "reference_geometry_offset_world_m": (
                reference_geometry_offset_world.astype(float).tolist()
            ),
            "effective_reference_sequential_speed": float(
                sequential_speed
            ),
            "sequential_reference_active": bool(
                sequential_reference_active
            ),
            "effective_reference_thrust_scale": float(thrust_scale),
            "effective_residual_scale": float(
                self.residual_scale
                * self.residual_scale_gates.get(gate_index, 1.0)
            ),
            "residual_phase_window": (
                list(phase_window) if phase_window is not None else None
            ),
            "residual_phase_active": bool(residual_active),
            "residual_schedule_active": bool(
                self.ppo_residual_schedule is not None
                and self.probe_arm == "candidate"
            ),
            "residual_actor_source": (
                "secondary_ppo" if secondary_actor_active else "primary"
            ),
            "probe_arm": self.probe_arm,
            "residual_schedule_gate": int(event_gate_index),
            "residual_schedule_phase": float(schedule_phase),
            "residual_schedule_correction": (
                schedule_correction.astype(float).tolist()
            ),
            "effective_teacher_blend": float(
                effective_teacher_blend
                if not line_backbone_active else self.teacher_blend
            ),
            "actor_speed_governor": float(
                actor_speed_governor
                if not line_backbone_active else 0.0
            ),
            "trajectory_blend_active": bool(trajectory_blend_active),
            "effective_trajectory_blend": float(
                effective_trajectory_blend
            ),
            "line_backbone_active": bool(line_backbone_active),
            "macro_exploration_gate": (
                int(self.episode_macro_gate)
                if self.episode_macro_gate is not None else -1
            ),
            "macro_lateral_residual": float(macro_lateral_residual),
            "raw_lateral_feedback": float(raw_lateral_feedback),
            "clipped_lateral_feedback": clipped_lateral_feedback,
            "lateral_feedback_saturated": bool(
                abs(float(raw_lateral_feedback))
                >= lateral_feedback_limit - 1e-6
            ),
            "raw_longitudinal_feedback": float(
                raw_longitudinal_feedback
            ),
            "effective_longitudinal_position_gain": float(
                longitudinal_position_gain
            ),
            "effective_longitudinal_velocity_gain": float(
                longitudinal_velocity_gain
            ),
            "clipped_longitudinal_feedback": (
                clipped_longitudinal_feedback
            ),
            "longitudinal_feedback_saturated": bool(
                abs(float(raw_longitudinal_feedback)) >= 0.20 - 1e-6
            ),
            "selected_reference_row": (
                float(self.reference_cursor)
                if line_backbone_active
                else float(np.sum(action_rows * weights))
            ),
            "reference_segment_start": gate_start,
            "reference_segment_end": gate_end,
            "lateral_position_error_m": float(
                position_error_reference[1]
            ),
            "lateral_velocity_error_mps": float(
                velocity_error_reference[1]
            ),
        }
        return (
            selected[0].cpu().numpy().astype(np.float32),
            residual[0].cpu().numpy().astype(np.float32),
            residual_mean[0].cpu().numpy().astype(np.float32),
        )

    def add_episode(self, transitions: list[dict]) -> None:
        """Add n-step rows and promote every passed-gate approach for SIL."""
        finished_episode = bool(
            transitions
            and float(transitions[-1]["done"]) > 0.5
            and float(transitions[-1]["reward"]) > 300.0
        )
        successful_rows: set[int] = set()
        for end, transition in enumerate(transitions):
            next_gate = int(np.argmax(
                transition["next_observation"][34:51]
            ))
            gate = int(transition["gate_index"])
            if next_gate > gate:
                start = end
                while (
                    start > 0
                    and end - start < 89
                    and int(transitions[start - 1]["gate_index"]) == gate
                ):
                    start -= 1
                successful_rows.update(range(start, end + 1))

        for row, transition in enumerate(transitions):
            reward = 0.0
            discount = 1.0
            final = transition
            for offset in range(self.n_step):
                candidate = row + offset
                if candidate >= len(transitions):
                    break
                final = transitions[candidate]
                reward += discount * float(final["reward"])
                discount *= self.gamma
                if final["done"]:
                    break
            # Keep the final two seconds before every failure prominent.
            tail_boost = 4.0 if row >= len(transitions) - 60 else 1.0
            success_boost = 4.0 if row in successful_rows else 1.0
            finish_boost = (
                self.finish_replay_boost if finished_episode else 1.0
            )
            self.live.add(
                transition["observation"],
                transition["action"],
                reward,
                final["next_observation"],
                final["done"],
                transition["gate_index"],
                discount=discount,
                priority=tail_boost
                * success_boost
                * finish_boost
                * (1.0 + abs(float(reward))),
                is_demo=finished_episode or row in successful_rows,
            )

    def load_episode_directory(self, path: Path) -> tuple[int, int]:
        """Reconstruct replay from durable per-episode NPZ flight logs."""
        episodes = 0
        transitions_loaded = 0
        for episode_path in sorted(path.glob("episode_*.npz")):
            payload = np.load(episode_path, allow_pickle=False)
            rewards = np.asarray(payload["reward"], np.float32)
            # Older runs marked a valid finish stale because the simulator
            # stops its clock/IMU after the official final-gate event.
            inferred_finish = bool(
                len(rewards)
                and rewards[-1] > 300.0
                and float(np.asarray(payload["done"])[-1]) > 0.5
            )
            if (
                "timing_healthy" in payload
                and not bool(np.asarray(payload["timing_healthy"]).all())
                and not inferred_finish
            ):
                continue
            rows = []
            for row in range(len(rewards)):
                observation = np.asarray(
                    payload["observation"][row], np.float32
                )
                rows.append({
                    "observation": observation,
                    "action": np.asarray(
                        payload["action"][row], np.float32
                    ),
                    "reward": float(rewards[row]),
                    "next_observation": np.asarray(
                        payload["next_observation"][row], np.float32
                    ),
                    "done": float(payload["done"][row]),
                    # The action was selected from the pre-step observation.
                    "gate_index": int(np.argmax(observation[34:51])),
                })
            if rows:
                self.add_episode(rows)
                episodes += 1
                transitions_loaded += len(rows)
        return episodes, transitions_loaded

    @staticmethod
    def _merge(
        first: dict[str, np.ndarray],
        second: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return {
            key: np.concatenate([first[key], second[key]], axis=0)
            for key in first
        }

    def update(self, steps: int, batch_size: int = 256) -> dict:
        report = {}
        for _ in range(steps):
            post_warmup = self.updates >= self.actor_warmup_updates
            current_actor_lr = (
                self.actor_lr if post_warmup else self.warmup_actor_lr
            )
            train_actor = current_actor_lr > 0.0
            for group in self.actor_optimizer.param_groups:
                group["lr"] = current_actor_lr
            live_count = batch_size // 2 if self.live.size >= 64 else 0
            _di, demo = self.demo.sample(
                batch_size - live_count,
                prioritized=True,
                gate_balanced=True,
            )
            if live_count:
                live_indices, live = self.live.sample(
                    live_count,
                    prioritized=True,
                    gate_balanced=True,
                )
                batch = self._merge(demo, live)
            else:
                live_indices = np.empty(0, int)
                batch = demo

            observation = torch.from_numpy(
                self.normalize(batch["observation"])
            ).to(self.device)
            action = torch.from_numpy(batch["action"]).to(self.device)
            reward = torch.from_numpy(
                batch["reward"][:, None] / self.return_scale
            ).to(self.device)
            next_observation = torch.from_numpy(
                self.normalize(batch["next_observation"])
            ).to(self.device)
            done = torch.from_numpy(batch["done"][:, None]).to(self.device)
            discount = torch.from_numpy(
                batch["discount"][:, None]
            ).to(self.device)
            batch_gate = np.asarray(batch["gate_index"], np.int16)
            if self.train_gate < 0:
                actor_rows = np.ones(len(batch_gate), dtype=bool)
                next_actor_rows = np.ones(len(batch_gate), dtype=bool)
            else:
                actor_rows = batch_gate == self.train_gate
                next_gate = np.argmax(
                    batch["next_observation"][:, 34:51],
                    axis=1,
                )
                next_actor_rows = next_gate == self.train_gate
            next_actor_mask = torch.from_numpy(
                next_actor_rows[:, None].astype(np.float32)
            ).to(self.device)

            with torch.no_grad():
                next_action, next_logp, _ = self.actor.sample(
                    next_observation
                )
                next_action = torch.clamp(
                    next_action,
                    -self.residual_output_clip,
                    self.residual_output_clip,
                )
                # The deployed curriculum emits exactly zero residual outside
                # the active gate. Bellman targets must obey that same action
                # support or the critic will value impossible future actions.
                next_action = next_action * next_actor_mask
                next_logp = next_logp * next_actor_mask
                tq1, tq2 = self.target_critic(
                    next_observation, next_action
                )
                target = reward + discount * (1.0 - done) * (
                    torch.minimum(tq1, tq2) - self.alpha * next_logp
                )
            q1, q2 = self.critic(observation, action)
            critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()

            active_count = int(np.count_nonzero(actor_rows))
            positive_advantage_count = 0
            if train_actor and active_count:
                active_mask = torch.from_numpy(actor_rows).to(self.device)
                active_observation = observation[active_mask]
                active_action = action[active_mask]
                active_is_demo = torch.from_numpy(
                    batch["is_demo"][actor_rows] > 0.5
                ).to(self.device)
                mean_action = self.actor.deterministic(active_observation)
                zero = torch.zeros((), device=self.device)
                if bool(active_is_demo.any()):
                    bc_loss = F.mse_loss(
                        mean_action[active_is_demo],
                        active_action[active_is_demo],
                    )
                else:
                    bc_loss = zero
                trust_loss = F.smooth_l1_loss(
                    mean_action,
                    torch.zeros_like(mean_action),
                    beta=0.05,
                )
                sac_loss = zero
                awr_loss = zero
                if self.actor_objective == "sac" and post_warmup:
                    sampled_action, logp, _ = self.actor.sample(
                        active_observation
                    )
                    sampled_action = torch.clamp(
                        sampled_action,
                        -self.residual_output_clip,
                        self.residual_output_clip,
                    )
                    aq1, aq2 = self.critic(
                        active_observation,
                        sampled_action,
                    )
                    sac_loss = (
                        self.alpha * logp - torch.minimum(aq1, aq2)
                    ).mean()
                elif self.actor_objective == "awr" and post_warmup:
                    live_rows = ~active_is_demo
                    if bool(live_rows.any()):
                        with torch.no_grad():
                            behavior_q1, behavior_q2 = self.critic(
                                active_observation[live_rows],
                                active_action[live_rows],
                            )
                            policy_action = torch.clamp(
                                mean_action[live_rows],
                                -self.residual_output_clip,
                                self.residual_output_clip,
                            )
                            policy_q1, policy_q2 = self.critic(
                                active_observation[live_rows],
                                policy_action,
                            )
                            advantage = (
                                torch.minimum(behavior_q1, behavior_q2)
                                - torch.minimum(policy_q1, policy_q2)
                            ).reshape(-1)
                            positive = advantage > 0.0
                            weights = torch.exp(
                                torch.clamp(
                                    advantage / self.awr_temperature,
                                    max=np.log(self.awr_max_weight),
                                )
                            )
                            weights = weights * positive
                        positive_advantage_count = int(positive.sum())
                        if positive_advantage_count:
                            regression = (
                                mean_action[live_rows]
                                - active_action[live_rows]
                            ).square().mean(dim=1)
                            awr_loss = (
                                weights * regression
                            ).sum() / weights.sum().clamp_min(1e-6)
                actor_loss = (
                    self.sac_actor_weight * sac_loss
                    + self.bc_weight * bc_loss
                    + self.awr_weight * awr_loss
                    + self.trust_weight * trust_loss
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
                self.actor_optimizer.step()
            else:
                zero = torch.zeros((), device=self.device)
                sac_loss = zero
                awr_loss = zero
                bc_loss = zero
                trust_loss = zero
                actor_loss = zero
                mean_action = torch.zeros(
                    (1, ACT_DIM),
                    device=self.device,
                )

            tau = 0.005
            with torch.no_grad():
                for target_parameter, parameter in zip(
                    self.target_critic.parameters(),
                    self.critic.parameters(),
                ):
                    target_parameter.lerp_(parameter, tau)
            if live_count:
                live_td = (
                    target[-live_count:]
                    - 0.5 * (
                        q1[-live_count:] + q2[-live_count:]
                    )
                )
                self.live.priority[live_indices] = (
                    (
                        live_td.abs()
                        + self.positive_td_priority
                        * torch.relu(live_td)
                    ).detach().cpu().numpy().reshape(-1)
                    + 1e-3
                )
            self.updates += 1
            report = {
                "critic_loss": float(critic_loss.detach()),
                "actor_loss": float(actor_loss.detach()),
                "sac_loss": float(sac_loss.detach()),
                "awr_loss": float(awr_loss.detach()),
                "bc_loss": float(bc_loss.detach()),
                "trust_loss": float(trust_loss.detach()),
                "mean_action_abs": float(mean_action.abs().mean().detach()),
                "actor_training": bool(train_actor),
                "actor_post_warmup": bool(post_warmup),
                "actor_lr": current_actor_lr,
                "active_gate_samples": active_count,
                "positive_advantage_samples": positive_advantage_count,
                "updates": self.updates,
            }
        self.inference_actor.load_state_dict(self.actor.state_dict())
        return report

    def checkpoint(self) -> dict:
        payload = {
            "kind": "vq2_residual_sac",
            "teacher_actor": self.teacher.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "observation_mean": self.observation_mean,
            "observation_std": self.observation_std,
            "return_scale": self.return_scale,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "bc_weight": self.bc_weight,
            "sac_actor_weight": self.sac_actor_weight,
            "trust_weight": self.trust_weight,
            "teacher_blend": self.teacher_blend,
            "teacher_blends": dict(self.teacher_blends),
            "actor_speed_limits": dict(self.actor_speed_limits),
            "actor_speed_governor_margin": (
                self.actor_speed_governor_margin
            ),
            "residual_scale": self.residual_scale,
            "residual_output_clip": self.residual_output_clip,
            "macro_exploration_gates": list(
                self.macro_exploration_gates
            ),
            "macro_lateral_residual_scale": (
                self.macro_lateral_residual_scale
            ),
            "actor_warmup_updates": self.actor_warmup_updates,
            "actor_lr": self.actor_lr,
            "warmup_actor_lr": self.warmup_actor_lr,
            "longitudinal_position_gain": self.longitudinal_position_gain,
            "longitudinal_velocity_gain": self.longitudinal_velocity_gain,
            "longitudinal_position_gains": dict(
                self.longitudinal_position_gains
            ),
            "longitudinal_velocity_gains": dict(
                self.longitudinal_velocity_gains
            ),
            "lateral_position_gain": self.lateral_position_gain,
            "lateral_velocity_gain": self.lateral_velocity_gain,
            "lateral_gain_scales": dict(self.lateral_gain_scales),
            "lateral_feedback_limits": dict(
                self.lateral_feedback_limits
            ),
            "lateral_bias_gates": sorted(self.lateral_bias_gates),
            "lateral_action_bias": self.lateral_action_bias,
            "right_lateral_bias_gates": sorted(
                self.right_lateral_bias_gates
            ),
            "right_lateral_action_bias": self.right_lateral_action_bias,
            "extra_lateral_biases": dict(self.extra_lateral_biases),
            "reference_action_lead": self.reference_action_lead,
            "reference_action_leads": dict(
                self.reference_action_leads
            ),
            "reference_mode": self.reference_mode,
            "reference_feedback_scale": self.reference_feedback_scale,
            "reference_thrust_scale": self.reference_thrust_scale,
            "reference_thrust_scales": dict(
                self.reference_thrust_scales
            ),
            "reference_rate_scale": self.reference_rate_scale,
            "reference_rate_scales": dict(self.reference_rate_scales),
            "reference_velocity_scale": self.reference_velocity_scale,
            "reference_velocity_scales": dict(
                self.reference_velocity_scales
            ),
            "reference_lateral_offsets": dict(
                self.reference_lateral_offsets
            ),
            "reference_vertical_offsets": dict(
                self.reference_vertical_offsets
            ),
            "reference_sequential_speed": self.reference_sequential_speed,
            "reference_sequential_speeds": dict(
                self.reference_sequential_speeds
            ),
            "predictive_handoff_distances": dict(
                self.predictive_handoff_distances
            ),
            "gate_center_funnel_gates": sorted(
                self.gate_center_funnel_gates
            ),
            "vertical_bias_gates": sorted(self.vertical_bias_gates),
            "vertical_action_bias": self.vertical_action_bias,
            "extra_vertical_biases": dict(
                self.extra_vertical_biases
            ),
            "trajectory_blend": self.trajectory_blend,
            "trajectory_blend_gates": sorted(
                self.trajectory_blend_gates
            ),
            "trajectory_blends": dict(self.trajectory_blends),
            "trajectory_kp_scale": self.trajectory_kp_scale,
            "trajectory_kv_scale": self.trajectory_kv_scale,
            "trajectory_attitude_gain": self.trajectory_attitude_gain,
            "gate4_action_lead": self.gate4_action_lead,
            "reference_max_advance": self.reference_max_advance,
            "reference_max_retreat": self.reference_max_retreat,
            "train_gate": self.train_gate,
            "residual_gates": sorted(self.residual_gates),
            "residual_phase_windows": {
                str(gate): [start, end]
                for gate, (start, end) in self.residual_phase_windows.items()
            },
            "frozen_residual_gates": sorted(
                self.frozen_residual_profiles
            ),
            "frozen_action_gates": sorted(self.frozen_action_profiles),
            "actor_objective": self.actor_objective,
            "awr_weight": self.awr_weight,
            "awr_temperature": self.awr_temperature,
            "awr_max_weight": self.awr_max_weight,
            "positive_td_priority": self.positive_td_priority,
            "finish_replay_boost": self.finish_replay_boost,
            "n_step": self.n_step,
            "updates": self.updates,
            "live_replay_size": self.live.size,
        }
        if self.recurrent_teacher is not None:
            payload["recurrent_teacher_actor"] = (
                self.recurrent_teacher.state_dict()
            )
            payload["recurrent_teacher"] = {
                "hidden_dim": self.recurrent_teacher.hidden_dim,
            }
        payload.update(copy.deepcopy(self.counterfactual_checkpoint_payload))
        return payload

    def restore_training_state(self, payload: dict) -> None:
        """Restore the actor and the training state that drives its updates."""
        self.actor.load_state_dict(payload["actor"])
        self.inference_actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.target_critic.load_state_dict(payload["target_critic"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
        if "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])

    def restore_actor_state(self, payload: dict) -> None:
        """Rollback a candidate actor without erasing learned value/replay."""
        self.actor.load_state_dict(payload["actor"])
        self.inference_actor.load_state_dict(payload["actor"])
        if "actor_optimizer" in payload:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])


def write_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, separators=(",", ":")) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--random-seed", type=int, default=11,
        help="Seed for exploration, impulse plans, and learner sampling.",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--multigate-ab",
        action="store_true",
        help=(
            "Alternate baseline and AIGP_MULTIGATE=1 by episode inside one "
            "sim session. Episode summaries record the active arm."
        ),
    )
    parser.add_argument(
        "--poc-stop-after-gate",
        type=int,
        default=-1,
        help=(
            "Short-course collection mode: end and reset immediately after "
            "the official crossing event for this zero-based gate. The run "
            "is logged as poc_completed, not as a full-course finish."
        ),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Run every episode deterministically without replay or updates.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=10,
        help=(
            "Run an update-free deterministic evaluation every N episodes. "
            "Set to zero to disable periodic evaluation."
        ),
    )
    parser.add_argument("--mav-port", type=int, default=14550)
    parser.add_argument("--camera-port", type=int, default=5600)
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8899,
        help="Local live-localizer dashboard port; set to zero to disable.",
    )
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--vision-hz",
        type=float,
        default=10.0,
        help="Maximum asynchronous GateNet inference rate.",
    )
    parser.add_argument(
        "--vision-device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help=(
            "Device for dense V7/V10 GateNet inference. Crop GateNet and "
            "YOLO remain on the automatically selected accelerator."
        ),
    )
    parser.add_argument(
        "--max-vision-result-age",
        type=float,
        default=0.30,
        help=(
            "Maximum camera-to-consumption age for an asynchronous dense "
            "vision result. Delayed results are fused out of sequence and "
            "the buffered IMU is replayed to the present."
        ),
    )
    parser.add_argument(
        "--vision-process-isolation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run dense V7/V10 inference in a persistent subprocess so "
            "PyTorch cannot stall MAVLink reception or flight control."
        ),
    )
    parser.add_argument(
        "--vision-worker-threads",
        type=int,
        default=4,
        help="Intra-op threads reserved for the isolated GateNet worker.",
    )
    parser.add_argument(
        "--vision-worker-affinity",
        default="all",
        help=(
            "CPU mask for the isolated GateNet worker. Keep it disjoint from "
            "the real-time trainer mask and the simulator's busiest cores."
        ),
    )
    parser.add_argument(
        "--crop-tracker",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run the guarded CropGateNet tracker between dense GateNet fixes. "
            "Enabled by default for live flight because dense CPU inference "
            "alone leaves the EKF without a fresh landmark for too long."
        ),
    )
    parser.add_argument(
        "--crop-tracker-hz",
        type=float,
        default=10.0,
        help="Maximum guarded crop-tracker rate; zero disables the tracker.",
    )
    parser.add_argument(
        "--crop-tracker-gates",
        default="",
        help=(
            "Optional comma-separated target gates where crop tracking is "
            "allowed. Empty preserves the legacy all-gates behavior."
        ),
    )
    parser.add_argument(
        "--crop-direct-position-pins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use the guarded active-gate PnP position pin when CropGateNet "
            "returns all four matched inner corners."
        ),
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=2,
        help="CPU learner threads; keep low so simulator physics stays responsive.",
    )
    parser.add_argument(
        "--torch-interop-threads",
        type=int,
        default=1,
        help="PyTorch inter-op threads used by the CPU learner.",
    )
    parser.add_argument(
        "--cpu-affinity",
        default="all",
        help=(
            "Process CPU mask such as 0xFFC0. Use a mask disjoint from the "
            "simulator so learning and vision cannot stall simulator physics."
        ),
    )
    parser.add_argument("--updates-per-step", type=float, default=1.0)
    parser.add_argument("--max-updates-per-episode", type=int, default=1200)
    parser.add_argument(
        "--offline-critic-warmup-updates",
        type=int,
        default=0,
        help=(
            "Run this many critic-only replay updates before opening live "
            "simulator ports. The actor remains frozen, so a new success "
            "dataset can propagate value without changing episode-0 control."
        ),
    )
    parser.add_argument(
        "--offline-updates",
        type=int,
        default=0,
        help=(
            "Run this many normal critic+actor replay updates before live "
            "flight. Unlike critic warmup, this applies the configured "
            "actor objective and learning rate."
        ),
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help=(
            "Save the offline candidate and exit without opening MAVLink, "
            "camera, dashboard, or simulator ports."
        ),
    )
    parser.add_argument("--actor-lr", type=float, default=1e-5)
    parser.add_argument(
        "--warmup-actor-lr",
        type=float,
        default=0.0,
        help=(
            "Actor learning rate before --actor-warmup-updates. Use a tiny "
            "value for expert-only supervised maintenance, or zero to freeze."
        ),
    )
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--bc-weight", type=float, default=0.25)
    parser.add_argument("--sac-actor-weight", type=float, default=1.0)
    parser.add_argument("--trust-weight", type=float, default=0.25)
    parser.add_argument("--teacher-blend", type=float, default=0.90)
    parser.add_argument(
        "--teacher-blends", default="",
        help=(
            "Comma-separated gate:protected-reference blend overrides; "
            "0 selects the distilled actor and 1 selects the reference."
        ),
    )
    parser.add_argument(
        "--actor-speed-limits", default="",
        help=(
            "Comma-separated gate:m/s limits that smoothly hand actor "
            "authority back to the protected reference near overspeed."
        ),
    )
    parser.add_argument(
        "--actor-speed-governor-margin", type=float, default=1.0,
        help="Width in m/s of the smooth actor-to-reference governor ramp.",
    )
    parser.add_argument("--exploration-clip", type=float, default=0.08)
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument(
        "--macro-exploration-gates",
        default="",
        help=(
            "Comma-separated gates eligible for one coherent episode-level "
            "lateral residual perturbation. Exactly one listed gate is "
            "sampled on each non-evaluation episode."
        ),
    )
    parser.add_argument(
        "--macro-lateral-residual-scale",
        type=float,
        default=0.0,
        help=(
            "Magnitude of the sampled normalized lateral residual. With "
            "--residual-scale 0.25, 0.06 produces a measurable 0.015 "
            "full-action line change."
        ),
    )
    parser.add_argument(
        "--domain-impulse-probability", type=float, default=0.0,
        help="Probability of one bounded action impulse during an episode.",
    )
    parser.add_argument(
        "--domain-impulse-gates", default="0,1,2,3",
        help="Comma-separated gates eligible for DAgger collection impulses.",
    )
    parser.add_argument(
        "--domain-impulse-axes", default="0,1,3",
        help="Canonical action axes eligible for a random impulse.",
    )
    parser.add_argument(
        "--domain-impulse-amplitude", type=float, default=0.0,
        help="Maximum absolute canonical action added during the impulse.",
    )
    parser.add_argument(
        "--domain-impulse-duration-steps", type=int, default=4,
        help="Number of 30 Hz control steps in each impulse.",
    )
    parser.add_argument(
        "--domain-impulse-min-distance", type=float, default=4.0,
        help="Nearest gate-plane distance at which an impulse may start.",
    )
    parser.add_argument(
        "--domain-impulse-max-distance", type=float, default=8.0,
        help="Farthest gate-plane distance at which an impulse may start.",
    )
    parser.add_argument(
        "--residual-output-clip",
        type=float,
        default=1.0,
        help=(
            "Hard clip applied to normalized residual actions in collection, "
            "evaluation, and Bellman targets."
        ),
    )
    parser.add_argument(
        "--residual-scale-gates",
        default="",
        help=(
            "Comma-separated gate:multiplier overrides applied to the global "
            "residual scale, e.g. 0:0.5,1:0.5."
        ),
    )
    parser.add_argument(
        "--residual-phase-windows",
        default="",
        help=(
            "Comma-separated gate:start:end course-phase windows. The "
            "residual actor is exactly zero outside each configured window."
        ),
    )
    parser.add_argument(
        "--zero-actor-output",
        action="store_true",
        help=(
            "Zero the loaded residual actor's mean head while preserving its "
            "feature backbone and critic. Use when expanding a gate-local "
            "checkpoint to full-course authority so deterministic episode 0 "
            "exactly reproduces the reference controller."
        ),
    )
    parser.add_argument("--actor-warmup-updates", type=int, default=5000)
    parser.add_argument(
        "--actor-objective",
        choices=("sac", "awr"),
        default="sac",
        help=(
            "sac directly maximizes critic Q; awr imitates only live actions "
            "that outperform the critic's current policy expectation."
        ),
    )
    parser.add_argument("--awr-weight", type=float, default=1.0)
    parser.add_argument("--awr-temperature", type=float, default=0.10)
    parser.add_argument("--awr-max-weight", type=float, default=20.0)
    parser.add_argument(
        "--positive-td-priority",
        type=float,
        default=2.0,
        help=(
            "Extra replay-priority multiplier for positive signed TD error."
        ),
    )
    parser.add_argument(
        "--finish-replay-boost",
        type=float,
        default=8.0,
        help=(
            "Priority multiplier for every transition in a completed lap. "
            "Completed laps are also retained as SIL demonstrations."
        ),
    )
    parser.add_argument(
        "--allow-supervised-only-actor",
        action="store_true",
        help=(
            "Explicitly allow a post-warmup actor with no reward-driven "
            "objective. This is unsafe for optimization campaigns and exists "
            "only for deliberate behavior-cloning maintenance runs."
        ),
    )
    parser.add_argument(
        "--n-step",
        type=int,
        default=12,
        help="Number of rewards accumulated in each live Bellman target.",
    )
    parser.add_argument(
        "--replay-path",
        type=Path,
        default=None,
        help="Optional persisted live_replay.npz to restore.",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=None,
        help=(
            "Optional prior run directory whose episode_*.npz logs are "
            "reconstructed into replay when no replay archive exists."
        ),
    )
    parser.add_argument(
        "--replay-save-interval",
        type=int,
        default=10,
        help=(
            "Persist replay every N collection episodes; it is always saved "
            "again during a clean shutdown."
        ),
    )
    parser.add_argument("--guardrail-failures", type=int, default=5)
    parser.add_argument(
        "--longitudinal-position-gain", type=float, default=0.0,
        help=(
            "Bounded forward-axis position catch-up gain for compressed "
            "sequential demonstrations."
        ),
    )
    parser.add_argument(
        "--longitudinal-velocity-gain", type=float, default=0.0,
        help=(
            "Bounded forward-axis velocity catch-up gain for compressed "
            "sequential demonstrations."
        ),
    )
    parser.add_argument(
        "--longitudinal-position-gains", default="",
        help="Comma-separated gate:gain forward-position overrides.",
    )
    parser.add_argument(
        "--longitudinal-velocity-gains", default="",
        help="Comma-separated gate:gain forward-velocity overrides.",
    )
    parser.add_argument("--lateral-position-gain", type=float, default=0.08)
    parser.add_argument("--lateral-velocity-gain", type=float, default=0.04)
    parser.add_argument(
        "--special-lateral-gate",
        type=int,
        default=-1,
        help="Gate index receiving the optional lateral feedback multiplier.",
    )
    parser.add_argument(
        "--special-lateral-gain-scale",
        type=float,
        default=1.0,
        help="Lateral feedback multiplier at --special-lateral-gate.",
    )
    parser.add_argument(
        "--lateral-gain-scales",
        default="",
        help=(
            "Comma-separated gate:scale lateral-feedback overrides. "
            "Unlike --special-lateral-gate, this preserves independent "
            "tuning at multiple gates, for example 5:2.0,10:1.5."
        ),
    )
    parser.add_argument(
        "--lateral-feedback-limits",
        default="",
        help=(
            "Comma-separated gate:absolute-limit overrides for lateral "
            "feedback; unspecified gates retain the 0.20 safety limit."
        ),
    )
    parser.add_argument(
        "--lateral-bias-gates",
        default="",
        help="Comma-separated gates receiving --lateral-action-bias.",
    )
    parser.add_argument(
        "--lateral-action-bias",
        type=float,
        default=0.0,
        help="Canonical roll bias on --lateral-bias-gates.",
    )
    parser.add_argument(
        "--right-lateral-bias-gates",
        default="",
        help=(
            "Comma-separated gates receiving an independent anticipatory "
            "rightward roll bias."
        ),
    )
    parser.add_argument(
        "--right-lateral-action-bias",
        type=float,
        default=0.0,
        help="Canonical rightward roll bias on those gates.",
    )
    parser.add_argument(
        "--extra-lateral-biases",
        default="",
        help=(
            "Comma-separated gate:bias adjustments applied in addition to "
            "the shared lateral biases, for example 7:-0.02."
        ),
    )
    parser.add_argument("--vertical-position-gain", type=float, default=0.30)
    parser.add_argument("--vertical-velocity-gain", type=float, default=0.10)
    parser.add_argument(
        "--special-vertical-gate",
        type=int,
        default=-1,
        help="Gate index receiving the optional vertical feedback multiplier.",
    )
    parser.add_argument(
        "--special-vertical-gain-scale",
        type=float,
        default=1.0,
        help="Vertical feedback multiplier at --special-vertical-gate.",
    )
    parser.add_argument(
        "--vertical-bias-gates",
        default="",
        help="Comma-separated gates receiving --vertical-action-bias.",
    )
    parser.add_argument(
        "--vertical-action-bias",
        type=float,
        default=0.0,
        help="Canonical upward thrust bias on --vertical-bias-gates.",
    )
    parser.add_argument(
        "--extra-vertical-biases",
        default="",
        help=(
            "Comma-separated gate:bias vertical trims applied in addition "
            "to the shared vertical bias, for example 4:0.025."
        ),
    )
    parser.add_argument(
        "--trajectory-blend",
        type=float,
        default=0.0,
        help=(
            "Blend a full 3-D position/velocity tracker around the clean "
            "demonstration into the demonstrated action."
        ),
    )
    parser.add_argument(
        "--trajectory-blend-gates",
        default="",
        help=(
            "Optional comma-separated gates where trajectory blending is "
            "enabled. Empty keeps the historical all-gates behavior."
        ),
    )
    parser.add_argument(
        "--trajectory-blends",
        default="",
        help=(
            "Comma-separated gate:blend overrides. This permits aggressive "
            "trajectory tracking on straights without imposing it on every "
            "precision gate."
        ),
    )
    parser.add_argument(
        "--trajectory-kp-scale",
        type=float,
        default=1.0,
        help="Scale the trajectory tracker's 3-D position gains.",
    )
    parser.add_argument(
        "--trajectory-kv-scale",
        type=float,
        default=1.0,
        help="Scale the trajectory tracker's 3-D velocity damping gains.",
    )
    parser.add_argument(
        "--trajectory-attitude-gain",
        type=float,
        default=4.0,
        help="Attitude-error to desired-body-rate gain for trajectory tracking.",
    )
    parser.add_argument(
        "--reference-mode",
        choices=("nearest", "sequential"),
        default="nearest",
        help="Select demo rows by live state or replay each gate sequentially.",
    )
    parser.add_argument(
        "--reference-feedback-scale",
        type=float,
        default=1.0,
        help="Scale all feedback added around demonstrated actions.",
    )
    parser.add_argument(
        "--reference-thrust-scale",
        type=float,
        default=1.0,
        help="Scale demonstrated thrust around the measured 0.25 hover point.",
    )
    parser.add_argument(
        "--reference-thrust-scales",
        default="",
        help=(
            "Comma-separated gate:scale thrust overrides. This supports a "
            "measured speed curriculum on straight segments without "
            "globally accelerating turn-critical gates."
        ),
    )
    parser.add_argument(
        "--reference-rate-scale",
        type=float,
        default=1.0,
        help=(
            "Scale demonstrated body-rate feed-forward. Pair scale s with "
            "sequential speed s and hover-centered thrust scale s^2 for a "
            "dynamically time-compressed expert trajectory."
        ),
    )
    parser.add_argument(
        "--reference-rate-scales",
        default="",
        help="Comma-separated gate:scale body-rate overrides.",
    )
    parser.add_argument(
        "--reference-velocity-scale",
        type=float,
        default=1.0,
        help=(
            "Time-compress the tracked spatial line: desired velocity uses "
            "this factor and feed-forward acceleration uses its square. "
            "Only affects the --trajectory-blend branch."
        ),
    )
    parser.add_argument(
        "--reference-velocity-scales",
        default="",
        help=(
            "Comma-separated gate:scale overrides for controlled, local "
            "speed-frontier experiments, for example 10:1.1,11:1.2."
        ),
    )
    parser.add_argument(
        "--reference-lateral-offsets",
        default="",
        help=(
            "Comma-separated gate:metres crossing offsets along each "
            "course-oriented gate's local lateral axis. Offsets are "
            "interpolated continuously between gates."
        ),
    )
    parser.add_argument(
        "--reference-vertical-offsets",
        default="",
        help=(
            "Comma-separated gate:metres crossing offsets along each "
            "gate's local vertical axis, smoothly interpolated by segment."
        ),
    )
    parser.add_argument(
        "--reference-sequential-speed",
        type=float,
        default=1.0,
        help=(
            "Demo rows advanced per control step in sequential mode. Values "
            "above one time-compress the demonstrated trajectory."
        ),
    )
    parser.add_argument(
        "--reference-sequential-speeds",
        default="",
        help=(
            "Comma-separated gate:rows-per-step overrides for controlled "
            "segment-wise time compression."
        ),
    )
    parser.add_argument(
        "--predictive-handoff-distances",
        default="",
        help=(
            "Comma-separated gate:meters overrides. Within this distance "
            "of a gate center, control may target the following gate before "
            "the authoritative gate event arrives."
        ),
    )
    parser.add_argument(
        "--gate-center-funnel-gates", default="",
        help=(
            "Comma-separated gates eligible for center-funnel correction. "
            "Empty applies an enabled funnel to all gates."
        ),
    )
    parser.add_argument(
        "--gate-center-funnel-distance", type=float, default=0.0,
        help="Begin blending the reference crossing toward gate center here.",
    )
    parser.add_argument(
        "--gate-center-funnel-full-distance", type=float, default=1.5,
        help="Reach the configured center-funnel strength at this distance.",
    )
    parser.add_argument(
        "--gate-center-funnel-strength", type=float, default=1.0,
        help="Maximum 0--1 blend from demonstrated crossing to aperture center.",
    )
    parser.add_argument(
        "--direct-position-pins",
        action="store_true",
        help="Strongly pin EKF position from complete fused active-gate quads.",
    )
    parser.add_argument("--reference-action-lead", type=int, default=0)
    parser.add_argument("--gate4-action-lead", type=int, default=0)
    parser.add_argument(
        "--reference-action-leads",
        default="",
        help=(
            "Comma-separated gate:frames action-lead overrides. Led rows "
            "are clamped to the active gate's demonstration segment."
        ),
    )
    parser.add_argument(
        "--reference-max-advance",
        type=int,
        default=8,
        help=(
            "Maximum clean-demonstration frames the monotonic reference "
            "cursor may advance per control step."
        ),
    )
    parser.add_argument(
        "--reference-max-retreat",
        type=int,
        default=2,
        help=(
            "Maximum clean-demonstration frames the reference cursor may "
            "retreat per control step to recover from timing jitter."
        ),
    )
    parser.add_argument(
        "--train-gate",
        type=int,
        default=-1,
        help=(
            "Only deploy and train residual actions while targeting this "
            "gate. -1 retains all-gate residual behavior."
        ),
    )
    parser.add_argument(
        "--residual-gates",
        default="",
        help=(
            "Comma-separated gates where the learned residual is deployed. "
            "Training still uses --train-gate. Empty preserves the legacy "
            "single-gate behavior."
        ),
    )
    parser.add_argument(
        "--frozen-residual-episode",
        type=Path,
        default=None,
        help=(
            "Episode NPZ supplying successful residual sequences for solved "
            "gates during later-gate curricula."
        ),
    )
    parser.add_argument(
        "--frozen-residual-gates",
        default="",
        help="Comma-separated gates loaded from --frozen-residual-episode.",
    )
    parser.add_argument(
        "--frozen-action-episode",
        type=Path,
        default=None,
        help=(
            "Episode NPZ supplying successful full-action profiles for "
            "solved gates during a later-gate curriculum."
        ),
    )
    parser.add_argument(
        "--frozen-action-gates",
        default="",
        help="Comma-separated gates loaded from --frozen-action-episode.",
    )
    parser.add_argument(
        "--seed-best-gate",
        type=int,
        default=-1,
        help=(
            "Known historical gate reach of --seed-checkpoint. Prevents a "
            "weaker rollout from replacing a stronger seeded best."
        ),
    )
    parser.add_argument("--thrust-cap", type=float, default=0.52)
    parser.add_argument("--launch-assist", type=float, default=0.55)
    parser.add_argument("--launch-min-thrust", type=float, default=0.30)
    parser.add_argument("--speed-cap", type=float, default=12.0)
    parser.add_argument(
        "--overspeed-confirmation",
        type=float,
        default=0.20,
        help="Require speed above --speed-cap for this many simulator seconds.",
    )
    parser.add_argument(
        "--overspeed-hysteresis",
        type=float,
        default=0.50,
        help="Re-arm overspeed only this far below --speed-cap.",
    )
    parser.add_argument("--gate-bonus", type=float, default=25.0)
    parser.add_argument("--finish-bonus", type=float, default=600.0)
    parser.add_argument("--progress-scale", type=float, default=2.0)
    parser.add_argument(
        "--time-penalty-per-s",
        type=float,
        default=0.8,
        help=(
            "Per-second racing penalty. Increase for a speed-optimization "
            "campaign after a reliable finishing baseline exists."
        ),
    )
    parser.add_argument(
        "--collision-penalty",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--failed-run-penalty",
        type=float,
        default=60.0,
    )
    parser.add_argument(
        "--action-smoothness",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--official-countdown",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hold zero thrust until MAVLink race_start_ms is reached. "
            "Enabled by default to avoid an official early-start DQ."
        ),
    )
    parser.add_argument(
        "--official-release-margin-ms",
        type=float,
        default=-1.0,
        help=(
            "Predict the scheduled simulator start from pending race-status "
            "packets and release this many ms afterward. Exactly -1 waits "
            "for the first authoritative active packet; values below -1 "
            "intentionally release before the scheduled start."
        ),
    )
    parser.add_argument(
        "--max-tilt",
        type=float,
        default=80.0,
        help="Terminate before the simulator's upside-down auto-respawn.",
    )
    parser.add_argument(
        "--anchor-timeout",
        type=float,
        default=0.75,
        help=(
            "Seconds spent collecting the spawn vision/IMU anchor before "
            "flying during the countdown."
        ),
    )
    parser.add_argument(
        "--gate-timeout",
        type=float,
        default=4.5,
        help=(
            "Hard-reset after this many seconds of simulator time without "
            "an official gate-pass event."
        ),
    )
    parser.add_argument(
        "--gate-event-plane-correction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use each authoritative gate-pass event as a one-dimensional "
            "EKF position fix on the crossed gate plane."
        ),
    )
    parser.add_argument(
        "--gate-event-forward-offset",
        type=float,
        default=0.0,
        help=(
            "Additional metres beyond the timestamp-propagated crossed "
            "plane assigned at event receipt."
        ),
    )
    parser.add_argument(
        "--max-camera-age",
        type=float,
        default=0.20,
        help="Terminate and quarantine an episode if raw camera packets get older.",
    )
    parser.add_argument(
        "--max-imu-age",
        type=float,
        default=0.10,
        help="Terminate and quarantine an episode if raw IMU packets get older.",
    )
    parser.add_argument(
        "--max-transition-sim-time",
        type=float,
        default=0.30,
        help="Quarantine transitions whose action was held this long in sim time.",
    )
    parser.add_argument(
        "--timing-failure-limit",
        type=int,
        default=2,
        help=(
            "After this many consecutive timing-unhealthy episodes, quarantine "
            "them and continue through the normal verified MAVLink drone reset."
        ),
    )
    parser.add_argument(
        "--max-episode-sim-step-p95",
        type=float,
        default=0.055,
        help=(
            "Mark an episode timing-unhealthy when its simulator-step p95 "
            "exceeds this many simulator seconds. Zero disables the check."
        ),
    )
    parser.add_argument(
        "--max-episode-sim-step-max",
        type=float,
        default=0.30,
        help=(
            "Mark an episode timing-unhealthy when any simulator step "
            "exceeds this many simulator seconds. Zero disables the check."
        ),
    )
    parser.add_argument(
        "--max-episode-step-p95-ms",
        type=float,
        default=60.0,
        help=(
            "Mark an episode timing-unhealthy when control-loop wall-time "
            "p95 exceeds this value. Zero disables the check."
        ),
    )
    parser.add_argument(
        "--seed-checkpoint",
        type=Path,
        default=REPO / "data/models/vq2_bc_seed.pt",
    )
    parser.add_argument(
        "--ppo-residual-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional fastsim PPO checkpoint. Its actor and observation "
            "normalizer replace only the bounded residual policy; the live "
            "reference controller, localizer, and reset logic are unchanged."
        ),
    )
    parser.add_argument(
        "--secondary-ppo-residual-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional second fastsim PPO checkpoint used only on "
            "--secondary-ppo-residual-gates. This preserves a proven primary "
            "actor while routing selected course segments to a specialized "
            "actor with its own observation normalizer."
        ),
    )
    parser.add_argument(
        "--secondary-ppo-residual-gates",
        default="",
        help=(
            "Comma-separated gates routed to "
            "--secondary-ppo-residual-checkpoint."
        ),
    )
    parser.add_argument(
        "--ppo-residual-schedule",
        type=Path,
        default=None,
        help=(
            "Optional optimize_vq2_residual_schedule.py result JSON. Its "
            "gate/segment-phase correction is added to the PPO residual in "
            "the same normalized coordinates used by fastsim."
        ),
    )
    parser.add_argument(
        "--interleave-protected-champion",
        action="store_true",
        help=(
            "Alternate deterministic episodes between the protected base "
            "controller and the PPO+schedule candidate in one session."
        ),
    )
    parser.add_argument(
        "--interleave-champion-demo",
        type=Path,
        default=None,
        help=(
            "Proven reference demo used by protected champion episodes. "
            "Candidate episodes continue to use --demo."
        ),
    )
    parser.add_argument(
        "--interleave-champion-config",
        type=Path,
        default=None,
        help=(
            "Full config.json for the protected controller arm. Every "
            "action-affecting reference gain, scale, bias, offset, lead, "
            "handoff, and funnel is switched in-process between arms."
        ),
    )
    parser.add_argument(
        "--probe-arm-sequence",
        default="protected_champion,candidate",
        help=(
            "Comma-separated in-process A/B arm cycle. Supported entries "
            "are candidate and protected_champion."
        ),
    )
    parser.add_argument(
        "--demo",
        type=Path,
        default=REPO / "data/vq2_sac_clean_demo.npz",
    )
    parser.add_argument(
        "--line",
        type=Path,
        default=None,
        help=(
            "Optional lineopt *_best.npz. Uses the exact live-validated "
            "FlatRefController as the SAC baseline; the actor remains a "
            "bounded residual policy."
        ),
    )
    parser.add_argument(
        "--line-model",
        type=Path,
        default=REPO / "data/fastsim_model_v2.json",
    )
    parser.add_argument("--line-speed-cap", type=float, default=8.0)
    parser.add_argument("--line-clearance", type=float, default=0.25)
    parser.add_argument(
        "--map",
        type=Path,
        default=REPO /
        "data/vq2_runtime_map_gift_v11sparse10hz_ep13.json",
    )
    parser.add_argument(
        "--primary",
        type=Path,
        default=REPO / "data/models/gatenet_v7_best.pt",
    )
    parser.add_argument(
        "--refiner",
        type=Path,
        default=REPO / "data/models/gatenet_v10strict_ep0.pt",
    )
    parser.add_argument(
        "--gate-primary",
        type=Path,
        default=None,
        help="Optional dense primary used only on --gate-primary-gates.",
    )
    parser.add_argument(
        "--gate-primary-gates",
        default="",
        help="Comma-separated gates that use --gate-primary.",
    )
    parser.add_argument(
        "--crop",
        type=Path,
        default=REPO / "data/models/crop_gatenet_v11crop_ep13.pt",
    )
    parser.add_argument(
        "--proposal",
        type=Path,
        default=REPO / "data/models/gatepose_v5vq2b_best.pt",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=REPO / "data/calib/calib.json",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path("D:/ai-gp/training/vq2_sac_runs")
            if Path("D:/").exists()
            else REPO / "data/vq2_sac_runs"
        ),
    )
    parser.add_argument(
        "--record-root",
        type=Path,
        default=(
            Path("D:/ai-gp/raw_sessions")
            if Path("D:/").exists()
            else REPO / "data/raw_sessions"
        ),
        help=(
            "Root for raw JPEG camera frames, raw MAVLink packets, and exact "
            "transmitted commands."
        ),
    )
    parser.add_argument(
        "--full-recording",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously archive all raw camera/MAVLink/control streams.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    probe_arm_sequence = tuple(
        value.strip()
        for value in str(args.probe_arm_sequence).split(",")
        if value.strip()
    )
    unknown_probe_arms = set(probe_arm_sequence) - {
        "candidate", "protected_champion"
    }
    if not probe_arm_sequence or unknown_probe_arms:
        raise ValueError(
            "--probe-arm-sequence must contain only candidate and "
            f"protected_champion; got {probe_arm_sequence}"
        )
    if (
        args.interleave_protected_champion
        and args.interleave_champion_config is None
    ):
        raise ValueError(
            "--interleave-protected-champion requires "
            "--interleave-champion-config so the control arm is exact"
        )
    reward_actor_weight = (
        args.awr_weight
        if args.actor_objective == "awr"
        else args.sac_actor_weight
    )
    if (
        not args.eval_only
        and args.actor_lr > 0.0
        and reward_actor_weight <= 0.0
        and not args.allow_supervised_only_actor
    ):
        raise ValueError(
            f"{args.actor_objective.upper()} actor objective has zero weight; "
            "the critic would learn but reward could not improve the actor. "
            "Set the corresponding reward weight above zero, freeze the "
            "actor with --actor-lr 0, or explicitly pass "
            "--allow-supervised-only-actor."
        )
    if args.crop_tracker_hz < 0.0:
        raise ValueError("--crop-tracker-hz must be non-negative")
    single_instance_guard = acquire_single_instance_guard()
    affinity = set_process_cpu_affinity(args.cpu_affinity)
    if affinity is not None:
        print(
            f"Trainer CPU affinity locked to 0x{affinity:X}",
            flush=True,
        )
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, int(args.torch_interop_threads)))
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    learner = VQ2SACLearner(
        args.seed_checkpoint,
        args.demo,
        line_path=args.line,
        line_map_path=args.map,
        line_model_path=args.line_model,
        line_speed_cap=args.line_speed_cap,
        line_clearance=args.line_clearance,
        device=args.device,
        actor_lr=args.actor_lr,
        warmup_actor_lr=args.warmup_actor_lr,
        critic_lr=args.critic_lr,
        alpha=args.alpha,
        bc_weight=args.bc_weight,
        sac_actor_weight=args.sac_actor_weight,
        trust_weight=args.trust_weight,
        teacher_blend=args.teacher_blend,
        teacher_blends=parse_gate_value_pairs(args.teacher_blends),
        actor_speed_limits=parse_gate_value_pairs(args.actor_speed_limits),
        actor_speed_governor_margin=args.actor_speed_governor_margin,
        residual_scale=args.residual_scale,
        residual_scale_gates=parse_gate_value_pairs(
            args.residual_scale_gates
        ),
        residual_output_clip=args.residual_output_clip,
        macro_exploration_gates=tuple(
            int(value.strip())
            for value in args.macro_exploration_gates.split(",")
            if value.strip()
        ),
        macro_lateral_residual_scale=args.macro_lateral_residual_scale,
        zero_actor_output=args.zero_actor_output,
        actor_warmup_updates=args.actor_warmup_updates,
        longitudinal_position_gain=args.longitudinal_position_gain,
        longitudinal_velocity_gain=args.longitudinal_velocity_gain,
        longitudinal_position_gains=parse_gate_value_pairs(
            args.longitudinal_position_gains,
            value_type=float,
        ),
        longitudinal_velocity_gains=parse_gate_value_pairs(
            args.longitudinal_velocity_gains,
            value_type=float,
        ),
        lateral_position_gain=args.lateral_position_gain,
        lateral_velocity_gain=args.lateral_velocity_gain,
        special_lateral_gate=args.special_lateral_gate,
        special_lateral_gain_scale=args.special_lateral_gain_scale,
        lateral_gain_scales=parse_gate_value_pairs(
            args.lateral_gain_scales
        ),
        lateral_feedback_limits=parse_gate_value_pairs(
            args.lateral_feedback_limits
        ),
        lateral_bias_gates=tuple(
            int(value.strip())
            for value in args.lateral_bias_gates.split(",")
            if value.strip()
        ),
        lateral_action_bias=args.lateral_action_bias,
        right_lateral_bias_gates=tuple(
            int(value.strip())
            for value in args.right_lateral_bias_gates.split(",")
            if value.strip()
        ),
        right_lateral_action_bias=args.right_lateral_action_bias,
        extra_lateral_biases=tuple(
            (
                int(item.split(":", 1)[0].strip()),
                float(item.split(":", 1)[1].strip()),
            )
            for item in args.extra_lateral_biases.split(",")
            if item.strip()
        ),
        vertical_position_gain=args.vertical_position_gain,
        vertical_velocity_gain=args.vertical_velocity_gain,
        special_vertical_gate=args.special_vertical_gate,
        special_vertical_gain_scale=args.special_vertical_gain_scale,
        vertical_bias_gates=tuple(
            int(value.strip())
            for value in args.vertical_bias_gates.split(",")
            if value.strip()
        ),
        vertical_action_bias=args.vertical_action_bias,
        extra_vertical_biases=parse_gate_value_pairs(
            args.extra_vertical_biases,
            value_type=float,
        ),
        trajectory_blend=args.trajectory_blend,
        trajectory_blend_gates=tuple(
            int(value.strip())
            for value in args.trajectory_blend_gates.split(",")
            if value.strip()
        ),
        trajectory_blends=parse_gate_value_pairs(
            args.trajectory_blends,
            value_type=float,
        ),
        trajectory_kp_scale=args.trajectory_kp_scale,
        trajectory_kv_scale=args.trajectory_kv_scale,
        trajectory_attitude_gain=args.trajectory_attitude_gain,
        reference_mode=args.reference_mode,
        reference_feedback_scale=args.reference_feedback_scale,
        reference_thrust_scale=args.reference_thrust_scale,
        reference_thrust_scales=parse_gate_value_pairs(
            args.reference_thrust_scales,
            value_type=float,
        ),
        reference_rate_scale=args.reference_rate_scale,
        reference_rate_scales=parse_gate_value_pairs(
            args.reference_rate_scales,
            value_type=float,
        ),
        reference_velocity_scale=args.reference_velocity_scale,
        reference_velocity_scales=parse_gate_value_pairs(
            args.reference_velocity_scales,
            value_type=float,
        ),
        reference_lateral_offsets=parse_gate_value_pairs(
            args.reference_lateral_offsets,
            value_type=float,
        ),
        reference_vertical_offsets=parse_gate_value_pairs(
            args.reference_vertical_offsets,
            value_type=float,
        ),
        reference_sequential_speed=args.reference_sequential_speed,
        reference_sequential_speeds=parse_gate_value_pairs(
            args.reference_sequential_speeds,
            value_type=float,
        ),
        predictive_handoff_distances=parse_gate_value_pairs(
            args.predictive_handoff_distances,
            value_type=float,
        ),
        gate_center_funnel_gates=tuple(
            int(value.strip())
            for value in args.gate_center_funnel_gates.split(",")
            if value.strip()
        ),
        gate_center_funnel_distance=args.gate_center_funnel_distance,
        gate_center_funnel_full_distance=(
            args.gate_center_funnel_full_distance
        ),
        gate_center_funnel_strength=args.gate_center_funnel_strength,
        reference_action_lead=args.reference_action_lead,
        gate4_action_lead=args.gate4_action_lead,
        reference_action_leads=parse_gate_value_pairs(
            args.reference_action_leads,
            value_type=int,
        ),
        reference_max_advance=args.reference_max_advance,
        reference_max_retreat=args.reference_max_retreat,
        train_gate=args.train_gate,
        residual_gates=tuple(
            int(value.strip())
            for value in args.residual_gates.split(",")
            if value.strip()
        ),
        residual_phase_windows=parse_gate_phase_windows(
            args.residual_phase_windows
        ),
        frozen_residual_episode=args.frozen_residual_episode,
        frozen_residual_gates=tuple(
            int(value.strip())
            for value in args.frozen_residual_gates.split(",")
            if value.strip()
        ),
        frozen_action_episode=args.frozen_action_episode,
        frozen_action_gates=tuple(
            int(value.strip())
            for value in args.frozen_action_gates.split(",")
            if value.strip()
        ),
        actor_objective=args.actor_objective,
        awr_weight=args.awr_weight,
        awr_temperature=args.awr_temperature,
        awr_max_weight=args.awr_max_weight,
        positive_td_priority=args.positive_td_priority,
        finish_replay_boost=args.finish_replay_boost,
        n_step=args.n_step,
        ppo_residual_checkpoint=args.ppo_residual_checkpoint,
        secondary_ppo_residual_checkpoint=(
            args.secondary_ppo_residual_checkpoint
        ),
        secondary_ppo_residual_gates=tuple(
            int(value.strip())
            for value in args.secondary_ppo_residual_gates.split(",")
            if value.strip()
        ),
        ppo_residual_schedule=args.ppo_residual_schedule,
        champion_demo_path=args.interleave_champion_demo,
        champion_config_path=args.interleave_champion_config,
    )
    replay_source = args.replay_path
    if replay_source is None:
        automatic_replay = args.seed_checkpoint.parent / "live_replay.npz"
        if automatic_replay.exists():
            replay_source = automatic_replay
    if replay_source is not None and replay_source.exists():
        replay_rows = learner.live.load(
            replay_source,
            default_discount=learner.gamma ** learner.n_step,
        )
        print(
            f"restored {replay_rows} live replay rows from {replay_source}",
            flush=True,
        )
    if args.replay_dir is not None:
        replay_episodes, replay_rows = learner.load_episode_directory(
            args.replay_dir
        )
        print(
            f"merged {replay_rows} rows from {replay_episodes} "
            f"timing-healthy episodes in {args.replay_dir}",
            flush=True,
        )
    warmup_updates = max(0, int(args.offline_critic_warmup_updates))
    if warmup_updates:
        saved_actor_lr = learner.actor_lr
        saved_warmup_actor_lr = learner.warmup_actor_lr
        learner.actor_lr = 0.0
        learner.warmup_actor_lr = 0.0
        completed = 0
        while completed < warmup_updates:
            chunk = min(500, warmup_updates - completed)
            report = learner.update(chunk)
            completed += chunk
            print(
                f"offline critic warmup {completed}/{warmup_updates} "
                f"critic_loss={report.get('critic_loss', float('nan')):.4f}",
                flush=True,
            )
        learner.actor_lr = saved_actor_lr
        learner.warmup_actor_lr = saved_warmup_actor_lr
        torch.save(learner.checkpoint(), run_dir / "offline_warmup.pt")
    config = VQ2EnvConfig(
        control_hz=args.control_hz,
        gate_bonus=args.gate_bonus,
        finish_bonus=args.finish_bonus,
        progress_scale=args.progress_scale,
        time_penalty_per_s=args.time_penalty_per_s,
        collision_penalty=args.collision_penalty,
        failed_run_penalty=args.failed_run_penalty,
        action_smoothness=args.action_smoothness,
        thrust_cap=args.thrust_cap,
        launch_assist_s=args.launch_assist,
        launch_min_thrust=args.launch_min_thrust,
        speed_cap_mps=args.speed_cap,
        overspeed_confirmation_s=args.overspeed_confirmation,
        overspeed_hysteresis_mps=args.overspeed_hysteresis,
        wait_for_official_start=args.official_countdown,
        official_release_margin_ms=args.official_release_margin_ms,
        max_tilt_deg=args.max_tilt,
        anchor_timeout_s=args.anchor_timeout,
        gate_timeout_s=args.gate_timeout,
        gate_event_plane_correction=args.gate_event_plane_correction,
        gate_event_forward_offset_m=args.gate_event_forward_offset,
        max_camera_packet_age_s=args.max_camera_age,
        max_imu_packet_age_s=args.max_imu_age,
        max_transition_sim_s=args.max_transition_sim_time,
    )
    config_payload = {
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": asdict(config),
    }
    config_hash = hashlib.sha256(json.dumps(
        config_payload, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    candidate_schedule_hash = sha256_file(args.ppo_residual_schedule)
    actor_artifact_hash = sha256_file(args.ppo_residual_checkpoint)
    secondary_actor_artifact_hash = sha256_file(
        args.secondary_ppo_residual_checkpoint
    )
    candidate_reference_hash = sha256_file(args.demo)
    champion_reference_hash = sha256_file(args.interleave_champion_demo)
    champion_config_hash = sha256_file(args.interleave_champion_config)
    config_payload["identity"] = {
        "config_sha256": config_hash,
        "candidate_schedule_sha256": candidate_schedule_hash,
        "actor_sha256": actor_artifact_hash,
        "secondary_actor_sha256": secondary_actor_artifact_hash,
        "candidate_reference_sha256": candidate_reference_hash,
        "champion_reference_sha256": champion_reference_hash,
        "champion_config_sha256": champion_config_hash,
        "probe_arm_sequence": list(probe_arm_sequence),
    }
    (run_dir / "config.json").write_text(json.dumps(
        config_payload, indent=2
    ))

    offline_updates = max(0, int(args.offline_updates))
    if offline_updates:
        completed = 0
        while completed < offline_updates:
            chunk = min(500, offline_updates - completed)
            report = learner.update(chunk)
            completed += chunk
            print(
                f"offline actor+critic {completed}/{offline_updates} "
                f"critic_loss={report.get('critic_loss', float('nan')):.4f} "
                f"actor_loss={report.get('actor_loss', float('nan')):.4f} "
                f"positive_advantage="
                f"{report.get('positive_advantage_samples', 0)}",
                flush=True,
            )
        torch.save(learner.checkpoint(), run_dir / "offline_candidate.pt")
    if args.offline_only:
        torch.save(learner.checkpoint(), run_dir / "latest.pt")
        if learner.live.size:
            learner.live.save(run_dir / "live_replay.npz")
        print(
            f"offline-only complete; no simulator ports opened: {run_dir}",
            flush=True,
        )
        return 0

    mavlink = None
    vision = None
    environment = None
    dashboard = None
    recorder = None
    stopping = False
    stop_request_path = run_dir / "STOP"

    def stop_handler(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True
        if environment is not None:
            environment.shutdown_to_spawn()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        if args.full_recording:
            recorder = FullSessionRecorder(
                args.record_root,
                f"vq2_{run_name}",
                metadata={
                    "training_run_dir": str(run_dir),
                    "config": config_payload,
                },
            )
            (run_dir / "raw_archive_path.txt").write_text(
                str(recorder.dir)
            )
        mavlink = MavIO(
            port=args.mav_port,
            on_record=(
                recorder.on_mavlink_event
                if recorder is not None else None
            ),
        )
        vision = VisionRX(
            port=args.camera_port,
            on_frame=(
                recorder.on_frame if recorder is not None else None
            ),
        )
        localizer = LiveVQ2Localizer(
            mavlink=mavlink,
            vision=vision,
            map_path=args.map,
            primary_checkpoint=args.primary,
            refine_checkpoint=args.refiner,
            gate_primary_checkpoint=args.gate_primary,
            gate_primary_gates=tuple(
                int(value.strip())
                for value in args.gate_primary_gates.split(",")
                if value.strip()
            ),
            crop_checkpoint=args.crop,
            proposal_checkpoint=args.proposal,
            calibration_path=args.calibration,
            async_interval_s=(
                1.0 / args.vision_hz if args.vision_hz > 0.0 else 0.0
            ),
            max_async_result_age_s=args.max_vision_result_age,
            dense_device=args.vision_device,
            dense_process_isolation=args.vision_process_isolation,
            dense_worker_threads=args.vision_worker_threads,
            dense_worker_affinity=args.vision_worker_affinity,
            direct_position_pins=args.direct_position_pins,
            crop_direct_position_pins=args.crop_direct_position_pins,
            crop_track_enabled=(
                args.crop_tracker and args.crop_tracker_hz > 0.0
            ),
            crop_track_interval_s=(
                1.0 / args.crop_tracker_hz
                if args.crop_tracker_hz > 0.0 else 1.0
            ),
            crop_track_gates=(
                tuple(
                    int(value.strip())
                    for value in args.crop_tracker_gates.split(",")
                    if value.strip()
                )
                if args.crop_tracker_gates.strip()
                else None
            ),
        )
        environment = VQ2LiveEnv(mavlink, localizer, config)
        best_gate = int(args.seed_best_gate)
        best_finish_time = np.inf
        best_checkpoint = copy.deepcopy(learner.checkpoint())
        best_checkpoint["best_gate"] = best_gate
        torch.save(best_checkpoint, run_dir / "best.pt")
        early_failures = 0
        timing_failure_streak = 0
        exploration_clip = float(args.exploration_clip)
        domain_impulse_gates = tuple(
            int(value.strip())
            for value in args.domain_impulse_gates.split(",")
            if value.strip()
        )
        domain_impulse_axes = tuple(
            int(value.strip())
            for value in args.domain_impulse_axes.split(",")
            if value.strip() and 0 <= int(value.strip()) < ACT_DIM
        )

        for episode in range(args.episodes):
            if stop_request_path.exists():
                stopping = True
            if stopping:
                break
            if args.multigate_ab:
                multigate_active = bool(episode % 2)
                if multigate_active:
                    os.environ["AIGP_MULTIGATE"] = "1"
                else:
                    os.environ.pop("AIGP_MULTIGATE", None)
                # Position-only mode is explicitly experimental and must not
                # leak into the validated A/B candidate arm.
                os.environ.pop("AIGP_MULTIGATE_POSONLY", None)
                print(
                    f"episode {episode:04d} localization_arm="
                    f"{'multigate' if multigate_active else 'baseline'}",
                    flush=True,
                )
            else:
                multigate_active = (
                    os.environ.get("AIGP_MULTIGATE") == "1"
                )
            reset_attempt = 0
            while not stopping:
                try:
                    observation, reset_info = environment.reset()
                    break
                except Exception as error:
                    reset_attempt += 1
                    print(
                        f"episode {episode:04d} RESET RETRY "
                        f"{reset_attempt}: {error!r}",
                        flush=True,
                    )
                    environment.emergency_stop()
                    environment.localizer.stop_async()
                    if not mavlink.receiver_alive:
                        print(
                            "FATAL: MAVLink receiver is not alive; "
                            "aborting instead of looping resets",
                            flush=True,
                        )
                        stopping = True
                        break
                    if reset_attempt >= 3:
                        print(
                            "FATAL: simulator reset failed three times; "
                            "aborting instead of looping resets",
                            flush=True,
                        )
                        stopping = True
                        break
                    time.sleep(0.25)
            if stopping:
                break
            if dashboard is None and args.dashboard_port > 0:
                dashboard = VQ2Dashboard(
                    localizer,
                    [gate["pos"] for gate in localizer.gates[:17]],
                    port=args.dashboard_port,
                    history_path=(
                        Path(args.record_root).parent
                        / "g0g4_poc_dashboard_history.jsonl"
                        if args.poc_stop_after_gate >= 0
                        else run_dir / "dashboard_results.jsonl"
                    ),
                    poc_gate=4,
                    control_hz=args.control_hz,
                )
                dashboard.start()
                print(
                    f"live localizer dashboard: "
                    f"http://localhost:{args.dashboard_port}",
                    flush=True,
                )
            scheduled_evaluation = bool(
                args.eval_only
                or args.smoke
                or episode == 0
                or (
                    args.eval_interval > 0
                    and episode % args.eval_interval == 0
                )
            )
            deterministic = scheduled_evaluation
            probe_arm = (
                probe_arm_sequence[episode % len(probe_arm_sequence)]
                if args.interleave_protected_champion
                else "candidate"
            )
            learner.set_probe_arm(probe_arm)
            # In an interleaved validation the event-plane update is part of
            # the candidate, not shared session infrastructure.  Keep the
            # protected champion bit-for-bit on its historical localization
            # path so it remains a valid session-health control.
            environment.gate_event_plane_correction_enabled = bool(
                args.gate_event_plane_correction
                and (
                    not args.interleave_protected_champion
                    or probe_arm == "candidate"
                )
            )
            learner.begin_episode(explore=not deterministic)
            impulse_plan = None
            if (
                domain_impulse_gates
                and domain_impulse_axes
                and args.domain_impulse_amplitude > 0.0
                and np.random.random()
                < float(np.clip(args.domain_impulse_probability, 0.0, 1.0))
            ):
                impulse_plan = {
                    "gate": int(np.random.choice(domain_impulse_gates)),
                    "axis": int(np.random.choice(domain_impulse_axes)),
                    "value": float(
                        np.random.choice((-1.0, 1.0))
                        * args.domain_impulse_amplitude
                        * np.random.uniform(0.6, 1.0)
                    ),
                    "remaining": max(
                        1, int(args.domain_impulse_duration_steps)
                    ),
                    "triggered": False,
                }
            transitions = []
            episode_crossings = []
            episode_reward = 0.0
            episode_localizer_counts = dict(localizer.update_counts)
            started = time.time()
            last_recorded_debug_frame = None
            print(
                f"episode {episode:04d} START deterministic={deterministic} "
                f"probe_arm={probe_arm} "
                f"countdown_hold={reset_info['countdown_hold_s']:.3f}s "
                f"release={reset_info['release_sim_boot_ms']}ms/"
                f"{reset_info['race_start_ms']}ms "
                f"macro_gate={learner.episode_macro_gate} "
                f"macro_lateral="
                f"{learner.episode_macro_lateral_residual:+.3f} "
                f"anchor={reset_info['anchor']}",
                flush=True,
            )
            final_info = {}
            poc_completed = False
            while not stopping:
                if stop_request_path.exists():
                    print(
                        f"graceful stop requested by {stop_request_path}",
                        flush=True,
                    )
                    stopping = True
                    environment.shutdown_to_spawn()
                    break
                action_gate_index = int(np.argmax(observation[34:51]))
                action, residual_action, actor_mean = learner.action(
                    observation,
                    deterministic=deterministic,
                    exploration_clip=exploration_clip,
                )
                control_debug = dict(learner.last_control_debug)
                selected_action = np.asarray(action, np.float32).copy()
                teacher_action = learner.last_reference_action.copy()
                impulse_delta = np.zeros(ACT_DIM, np.float32)
                impulse_active = False
                if impulse_plan is not None:
                    plane_distance = float(control_debug.get(
                        "gate_plane_distance_m", np.inf
                    ))
                    if (
                        not impulse_plan["triggered"]
                        and int(control_debug.get(
                            "control_gate_index", -1
                        )) == impulse_plan["gate"]
                        and args.domain_impulse_min_distance
                        <= plane_distance
                        <= args.domain_impulse_max_distance
                    ):
                        impulse_plan["triggered"] = True
                    if (
                        impulse_plan["triggered"]
                        and impulse_plan["remaining"] > 0
                    ):
                        impulse_active = True
                        impulse_delta[impulse_plan["axis"]] = (
                            impulse_plan["value"]
                        )
                        action = np.clip(
                            selected_action + impulse_delta, -1.0, 1.0
                        ).astype(np.float32)
                        impulse_plan["remaining"] -= 1
                control_debug.update({
                    "domain_impulse_active": bool(impulse_active),
                    "domain_impulse_gate": int(
                        impulse_plan["gate"]
                        if impulse_plan is not None else -1
                    ),
                    "domain_impulse_axis": int(
                        impulse_plan["axis"]
                        if impulse_plan is not None else -1
                    ),
                    "domain_impulse_value": float(
                        impulse_plan["value"]
                        if impulse_plan is not None else 0.0
                    ),
                })
                next_observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )
                crossing_payload = {}
                for crossing_offset in range(
                    int(info.get("gates_passed", 0))
                ):
                    crossed_gate = action_gate_index + crossing_offset
                    if not 0 <= crossed_gate < N_RACE_GATES:
                        continue
                    crossing = {
                        "gate": crossed_gate,
                        "step": len(transitions),
                        **gate_crossing_offset(
                            localizer.gates[crossed_gate],
                            np.asarray(info["position"], float),
                        ),
                    }
                    episode_crossings.append(crossing)
                    if crossing_offset == 0:
                        crossing_payload = {
                            "crossing_gate": crossed_gate,
                            "crossing_lateral_m": crossing["lateral_m"],
                            "crossing_plane_m": crossing["plane_m"],
                            "crossing_vertical_m": crossing["vertical_m"],
                        }
                transitions.append({
                    "observation": observation.copy(),
                    "action": residual_action.copy(),
                    "wire_action": np.asarray(
                        info["action"], np.float32
                    ),
                    "teacher_action": teacher_action,
                    "domain_impulse": impulse_delta,
                    "actor_mean": actor_mean.copy(),
                    "reference_row": int(learner.reference_cursor),
                    "reward": reward,
                    "next_observation": next_observation.copy(),
                    "done": float(terminated),
                    # Label the state that selected the action, not the
                    # post-step target (which advances on a crossing).
                    "gate_index": action_gate_index,
                    "position": np.asarray(
                        info["position"], np.float32
                    ),
                    "timing_healthy": bool(
                        info.get("timing_healthy", True)
                    ),
                    "camera_age_s": float(info["camera_age_s"]),
                    "imu_age_s": float(info["imu_age_s"]),
                    "visual_age_s": float(info["visual_age_s"]),
                    "sim_step_s": float(info["sim_step_s"]),
                    "step_ms": float(info["step_ms"]),
                    "gates_passed": int(info["gates_passed"]),
                    **control_debug,
                })
                step_payload = {
                    "episode": episode,
                    "step": len(transitions) - 1,
                    "reward": reward,
                    **info,
                    **control_debug,
                    **crossing_payload,
                }
                write_jsonl(run_dir / "steps.jsonl", step_payload)
                observation = next_observation
                episode_reward += reward
                final_info = info
                if dashboard is not None:
                    dashboard.update({
                        **info,
                        "episode": episode,
                        "step": len(transitions) - 1,
                        "episode_reward": episode_reward,
                        "reference_row": int(learner.reference_cursor),
                        **control_debug,
                        **crossing_payload,
                        "velocity": environment.localizer.state(
                        ).velocity.tolist(),
                    })
                if (
                    recorder is not None
                    and localizer.last_frame_id is not None
                    and localizer.last_frame_id
                    != last_recorded_debug_frame
                ):
                    debug_image, debug_payload = (
                        localizer.dashboard_snapshot()
                    )
                    debug_frame_id = debug_payload.get(
                        "frame_id", localizer.last_frame_id
                    )
                    if (
                        debug_image is not None
                        and debug_frame_id is not None
                        and debug_frame_id
                        != last_recorded_debug_frame
                    ):
                        recorder.on_localizer_debug(
                            int(debug_frame_id),
                            time.time_ns(),
                            debug_image,
                            debug_payload,
                        )
                        last_recorded_debug_frame = int(debug_frame_id)
                if info["gates_passed"] or len(transitions) % 30 == 0:
                    print(
                        f"  step={len(transitions):4d} "
                        f"gate={info['target']:2d} "
                        f"speed={info['speed']:4.1f} "
                        f"sigma={info['position_sigma_m']:.2f} "
                        f"landmark_age={info['visual_age_s']:.2f} "
                        f"camera_age={info['camera_age_s']:.3f} "
                        f"imu_age={info['imu_age_s']:.3f} "
                        f"reward={episode_reward:+.1f}",
                        flush=True,
                    )
                if (
                    args.poc_stop_after_gate >= 0
                    and any(
                        int(row.get("gate", -1))
                        >= args.poc_stop_after_gate
                        for row in episode_crossings
                    )
                ):
                    poc_completed = True
                    break
                if terminated or truncated:
                    break

            if stopping and not transitions:
                break
            duration = time.time() - started
            try:
                environment.park_after_episode()
            except Exception as error:
                print(
                    f"POST_EPISODE_RESET_WARNING: {error!r}",
                    flush=True,
                )
            reached = int(final_info.get("target", 0))
            finished = bool(final_info.get("finished", False))
            camera_ages = np.asarray([
                row["camera_age_s"] for row in transitions
            ], np.float32)
            imu_ages = np.asarray([
                row["imu_age_s"] for row in transitions
            ], np.float32)
            sim_steps = np.asarray([
                row["sim_step_s"] for row in transitions
            ], np.float32)
            step_times = np.asarray([
                row["step_ms"] for row in transitions
            ], np.float32)
            sim_step_p95_s = float(np.percentile(sim_steps, 95))
            sim_step_max_s = float(np.max(sim_steps))
            step_p95_ms = float(np.percentile(step_times, 95))
            step_max_ms = float(np.max(step_times))
            timing_health_reasons = []
            if not all(row["timing_healthy"] for row in transitions):
                timing_health_reasons.append("transition_timing")
            if (
                args.max_episode_sim_step_p95 > 0.0
                and sim_step_p95_s > args.max_episode_sim_step_p95
            ):
                timing_health_reasons.append("sim_step_p95")
            if (
                args.max_episode_sim_step_max > 0.0
                and sim_step_max_s > args.max_episode_sim_step_max
            ):
                timing_health_reasons.append("sim_step_max")
            if (
                args.max_episode_step_p95_ms > 0.0
                and step_p95_ms > args.max_episode_step_p95_ms
            ):
                timing_health_reasons.append("control_step_p95")
            timing_healthy = bool(
                transitions and not timing_health_reasons
            )
            if timing_healthy:
                timing_failure_streak = 0
            else:
                timing_failure_streak += 1
            cut_metrics = gate10_cut_metrics(transitions)
            summary = {
                "episode": episode,
                "run_id": run_dir.name,
                "config_sha256": config_hash,
                "schedule_arm": probe_arm,
                "schedule_sha256": (
                    candidate_schedule_hash
                    if probe_arm == "candidate" else champion_config_hash
                ),
                "actor_sha256": (
                    actor_artifact_hash
                    if probe_arm == "candidate" else None
                ),
                "reference_sha256": (
                    candidate_reference_hash
                    if probe_arm == "candidate"
                    else champion_reference_hash or candidate_reference_hash
                ),
                "steps": len(transitions),
                "duration_s": duration,
                "official_elapsed_s": (
                    float(final_info["official_gate_time_s"])
                    if poc_completed
                    and final_info.get("official_gate_time_s") is not None
                    else None
                ),
                "reward": episode_reward,
                "gate_reached": reached,
                "finished": finished,
                "poc_completed": poc_completed,
                "failure": final_info.get("failure"),
                "deterministic": deterministic,
                "evaluation": scheduled_evaluation,
                "multigate": bool(multigate_active),
                "localization_arm": (
                    "multigate" if multigate_active else "baseline"
                ),
                "gate_event_plane_correction": bool(
                    environment.gate_event_plane_correction_enabled
                ),
                "countdown_hold_s": float(
                    reset_info["countdown_hold_s"]
                ),
                "race_start_ms": int(reset_info["race_start_ms"]),
                "release_sim_boot_ms": int(
                    reset_info["release_sim_boot_ms"]
                ),
                "predictive_release": bool(
                    reset_info.get("predictive_release", False)
                ),
                "exploration_clip": exploration_clip,
                "macro_exploration_gate": (
                    int(learner.episode_macro_gate)
                    if learner.episode_macro_gate is not None else -1
                ),
                "macro_lateral_residual": float(
                    learner.episode_macro_lateral_residual
                ),
                "domain_impulse": (
                    dict(impulse_plan) if impulse_plan is not None else None
                ),
                "live_replay_size": learner.live.size,
                "updates": learner.updates,
                "timing_healthy": timing_healthy,
                "timing_health_reasons": timing_health_reasons,
                "timing_failure_streak": timing_failure_streak,
                "camera_age_p95_s": float(np.percentile(
                    camera_ages, 95
                )),
                "camera_age_max_s": float(np.max(camera_ages)),
                "imu_age_p95_s": float(np.percentile(imu_ages, 95)),
                "imu_age_max_s": float(np.max(imu_ages)),
                "sim_step_p95_s": sim_step_p95_s,
                "sim_step_max_s": sim_step_max_s,
                "step_p95_ms": step_p95_ms,
                "step_max_ms": step_max_ms,
                "landmark_age_p50_s": float(np.percentile(
                    [
                        row["visual_age_s"]
                        for row in transitions
                    ],
                    50,
                )),
                "landmark_age_p90_s": float(np.percentile(
                    [
                        row["visual_age_s"]
                        for row in transitions
                    ],
                    90,
                )),
                "crop_track_updates": int(
                    localizer.update_counts.get("crop_track", 0)
                    - episode_localizer_counts.get("crop_track", 0)
                ),
                "crop_track_rejections": int(
                    localizer.update_counts.get("crop_track_none", 0)
                    - episode_localizer_counts.get("crop_track_none", 0)
                ),
                "vision_fusion_by_gate": (
                    localizer.vision_fusion_by_gate()
                ),
                "crossing_offsets": episode_crossings,
                "gate10_cleared": any(
                    row["gate"] == 10 for row in episode_crossings
                ),
                "gate11_cleared": any(
                    row["gate"] == 11 for row in episode_crossings
                ),
                "gate_control_metrics": gate_control_metrics(
                    transitions
                ),
                **cut_metrics,
            }
            write_jsonl(run_dir / "episodes.jsonl", summary)
            if dashboard is not None:
                dashboard.record_episode(summary)
            np.savez_compressed(
                run_dir / f"episode_{episode:04d}.npz",
                observation=np.asarray([
                    row["observation"] for row in transitions
                ], np.float32),
                action=np.asarray([
                    row["action"] for row in transitions
                ], np.float32),
                wire_action=np.asarray([
                    row["wire_action"] for row in transitions
                ], np.float32),
                teacher_action=np.asarray([
                    row["teacher_action"] for row in transitions
                ], np.float32),
                teacher_action_source=np.asarray("protected_reference"),
                domain_impulse=np.asarray([
                    row["domain_impulse"] for row in transitions
                ], np.float32),
                actor_mean=np.asarray([
                    row["actor_mean"] for row in transitions
                ], np.float32),
                macro_exploration_gate=np.asarray([
                    row["macro_exploration_gate"] for row in transitions
                ], np.int16),
                macro_lateral_residual=np.asarray([
                    row["macro_lateral_residual"] for row in transitions
                ], np.float32),
                reference_row=np.asarray([
                    row["reference_row"] for row in transitions
                ], np.int32),
                reward=np.asarray([
                    row["reward"] for row in transitions
                ], np.float32),
                next_observation=np.asarray([
                    row["next_observation"] for row in transitions
                ], np.float32),
                done=np.asarray([
                    row["done"] for row in transitions
                ], np.float32),
                gate_index=np.asarray([
                    row["gate_index"] for row in transitions
                ], np.int16),
                position=np.asarray([
                    row["position"] for row in transitions
                ], np.float32),
                gates_passed=np.asarray([
                    row["gates_passed"] for row in transitions
                ], np.int16),
                effective_lateral_gain_scale=np.asarray([
                    row["effective_lateral_gain_scale"]
                    for row in transitions
                ], np.float32),
                effective_reference_action_lead=np.asarray([
                    row["effective_reference_action_lead"]
                    for row in transitions
                ], np.int16),
                effective_reference_velocity_scale=np.asarray([
                    row["effective_reference_velocity_scale"]
                    for row in transitions
                ], np.float32),
                effective_reference_thrust_scale=np.asarray([
                    row["effective_reference_thrust_scale"]
                    for row in transitions
                ], np.float32),
                raw_lateral_feedback=np.asarray([
                    row["raw_lateral_feedback"] for row in transitions
                ], np.float32),
                clipped_lateral_feedback=np.asarray([
                    row["clipped_lateral_feedback"]
                    for row in transitions
                ], np.float32),
                lateral_feedback_saturated=np.asarray([
                    row["lateral_feedback_saturated"]
                    for row in transitions
                ], np.bool_),
                selected_reference_row=np.asarray([
                    row["selected_reference_row"]
                    for row in transitions
                ], np.int32),
                lateral_position_error_m=np.asarray([
                    row["lateral_position_error_m"]
                    for row in transitions
                ], np.float32),
                lateral_velocity_error_mps=np.asarray([
                    row["lateral_velocity_error_mps"]
                    for row in transitions
                ], np.float32),
                timing_healthy=np.asarray([
                    row["timing_healthy"] for row in transitions
                ], np.bool_),
            )
            print(f"episode {episode:04d} END {summary}", flush=True)
            if (
                args.timing_failure_limit > 0
                and timing_failure_streak >= args.timing_failure_limit
            ):
                torch.save(learner.checkpoint(), run_dir / "latest.pt")
                print(
                    "TIMING_ABORT: consecutive unhealthy episodes reached "
                    f"{timing_failure_streak}; checkpointing and stopping "
                    "the trainer at spawn instead of crash-looping",
                    flush=True,
                )
                stopping = True

            if args.smoke or stopping:
                break
            if args.eval_only:
                continue
            if scheduled_evaluation:
                if reached > best_gate or (
                    finished and duration < best_finish_time
                ):
                    best_gate = reached
                    best_finish_time = (
                        duration if finished else best_finish_time
                    )
                    best_checkpoint = copy.deepcopy(learner.checkpoint())
                    best_checkpoint["best_gate"] = best_gate
                    torch.save(best_checkpoint, run_dir / "best.pt")
                    print(
                        f"EVAL: protected new best gate={best_gate}",
                        flush=True,
                    )
                if learner.train_gate >= 0:
                    if reached > learner.train_gate:
                        early_failures = 0
                    elif reached < learner.train_gate:
                        # In a gate-local curriculum the actor being judged
                        # never executed if the frozen prefix failed first.
                        # Treating that as evidence against the segment actor
                        # repeatedly erased useful gate-10 learning after
                        # unrelated gate-2/4/6 collisions.
                        print(
                            "EVAL: frozen-prefix failure before curriculum "
                            f"gate {learner.train_gate}; actor guardrail "
                            "unchanged",
                            flush=True,
                        )
                    else:
                        # The active segment was reached but not cleared, so
                        # this rollout contains direct evidence about it.
                        early_failures += 1
                        print(
                            "EVAL: curriculum gate was not cleared; "
                            f"guardrail failure {early_failures}/"
                            f"{args.guardrail_failures}",
                            flush=True,
                        )
                elif best_gate >= N_RACE_GATES:
                    if finished:
                        early_failures = 0
                    else:
                        early_failures += 1
                        print(
                            "EVAL: full-course candidate did not finish; "
                            f"guardrail failure {early_failures}/"
                            f"{args.guardrail_failures}",
                            flush=True,
                        )
                elif reached < 2:
                    early_failures += 1
                else:
                    early_failures = 0
                if (
                    early_failures >= args.guardrail_failures
                    and learner.updates < learner.actor_warmup_updates
                ):
                    # Nothing in the residual actor can regress while it is
                    # frozen. Keep the warmed critic and treat these as
                    # baseline evaluation variance.
                    early_failures = 0
                    print(
                        "GUARDRAIL: deferred during actor warmup",
                        flush=True,
                    )
                elif early_failures >= args.guardrail_failures:
                    # A noisy deterministic rollout can reject a candidate
                    # actor, but it must not erase critic learning or replay.
                    learner.restore_actor_state(best_checkpoint)
                    exploration_clip = max(
                        0.005, exploration_clip * 0.65
                    )
                    early_failures = 0
                    torch.save(
                        learner.checkpoint(), run_dir / "latest.pt"
                    )
                    print(
                        "GUARDRAIL: deterministic evaluations restored "
                        f"the protected actor only; exploration_clip="
                        f"{exploration_clip:.4f}",
                        flush=True,
                    )
                if timing_healthy and finished:
                    # A deterministic finish is the highest-value trajectory
                    # the live system can produce. Evaluation used to protect
                    # its checkpoint and then `continue`, silently discarding
                    # the entire lap from replay. Retain it as full-lap SIL so
                    # later updates repeatedly rehearse every verified gate.
                    learner.add_episode(transitions)
                    learner.live.save(run_dir / "live_replay.npz")
                    torch.save(
                        learner.checkpoint(), run_dir / "latest.pt"
                    )
                    print(
                        "SIL: retained deterministic full-course finish "
                        f"({len(transitions)} rows) in prioritized replay",
                        flush=True,
                    )
                elif (
                    args.replay_save_interval > 0
                    and episode % args.replay_save_interval == 0
                ):
                    # Evaluation returns before the normal collection save
                    # below. If eval and save intervals match, every scheduled
                    # save otherwise gets skipped and replay exists only in RAM
                    # until a clean shutdown.
                    learner.live.save(run_dir / "live_replay.npz")
                continue
            if not timing_healthy:
                print(
                    "QUARANTINE: skipped replay and gradient updates for "
                    "timing-unhealthy episode",
                    flush=True,
                )
                torch.save(learner.checkpoint(), run_dir / "latest.pt")
                continue
            learner.add_episode(transitions)
            if (
                args.replay_save_interval > 0
                and episode % args.replay_save_interval == 0
            ):
                learner.live.save(run_dir / "live_replay.npz")
            guardrail_restored = False

            update_steps = min(
                args.max_updates_per_episode,
                max(1, int(
                    len(transitions) * args.updates_per_step
                )),
            )
            if guardrail_restored:
                update_report = {
                    "skipped": True,
                    "reason": "guardrail_restore",
                    "updates": learner.updates,
                }
            else:
                update_report = learner.update(update_steps)
            write_jsonl(run_dir / "updates.jsonl", {
                "episode": episode,
                "update_steps": update_steps,
                **update_report,
            })
            torch.save(learner.checkpoint(), run_dir / "latest.pt")
            print(f"offline SAC {update_report}", flush=True)
            if (
                best_checkpoint.get("updates", 0)
                < learner.actor_warmup_updates
                <= learner.updates
            ):
                # Preserve the historical gate reach but promote the critic
                # that has now learned the residual-action coordinate system.
                best_checkpoint = copy.deepcopy(learner.checkpoint())
                best_checkpoint["best_gate"] = best_gate
                torch.save(best_checkpoint, run_dir / "best.pt")
                print(
                    "WARMUP: promoted zero-residual learner with trained "
                    "critic to protected best",
                    flush=True,
                )
    finally:
        torch.save(learner.checkpoint(), run_dir / "latest.pt")
        if learner.live.size:
            learner.live.save(run_dir / "live_replay.npz")
        if dashboard is not None:
            dashboard.close()
        if environment is not None:
            environment.shutdown_to_spawn()
            environment.localizer.close()
        if vision is not None:
            vision.close()
        if mavlink is not None:
            mavlink.close()
        if recorder is not None:
            recorder.close()
    print(f"run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
