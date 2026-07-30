"""Run behavior-cloned VQ2 policy and fine-tune it with offline-between-laps SAC.

The control loop never performs gradient work while the drone is flying.
Transitions are logged immediately, failed episodes are hard-terminal, and
SAC updates run only after the simulator has been stopped.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from aigp.mavlink_io import MavIO  # noqa: E402
from aigp.flight import RATE_CMD_GAIN, RATE_CMD_SIGN, RateController  # noqa: E402
from aigp.rl.sac import GaussianActor, TwinCritic  # noqa: E402
from aigp.rl.vq2_env import VQ2EnvConfig, VQ2LiveEnv  # noqa: E402
from aigp.rl.vq2_features import (  # noqa: E402
    ACT_DIM,
    N_RACE_GATES,
    OBS_DIM,
    WIRE_RATE_LIMIT,
)
from aigp.vision_io import VisionRX  # noqa: E402
from aigp.vq2_live_localizer import LiveVQ2Localizer  # noqa: E402
from aigp.vq2_dashboard import VQ2Dashboard  # noqa: E402


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
        device: str,
        actor_lr: float,
        warmup_actor_lr: float,
        critic_lr: float,
        alpha: float,
        bc_weight: float,
        sac_actor_weight: float,
        trust_weight: float,
        teacher_blend: float,
        residual_scale: float,
        residual_output_clip: float,
        actor_warmup_updates: int,
        lateral_position_gain: float,
        lateral_velocity_gain: float,
        special_lateral_gate: int,
        special_lateral_gain_scale: float,
        lateral_bias_gates: tuple[int, ...],
        lateral_action_bias: float,
        right_lateral_bias_gates: tuple[int, ...],
        right_lateral_action_bias: float,
        vertical_position_gain: float,
        vertical_velocity_gain: float,
        special_vertical_gate: int,
        special_vertical_gain_scale: float,
        vertical_bias_gates: tuple[int, ...],
        vertical_action_bias: float,
        trajectory_blend: float,
        reference_mode: str,
        reference_feedback_scale: float,
        reference_thrust_scale: float,
        reference_action_lead: int,
        gate4_action_lead: int,
        reference_max_advance: int,
        reference_max_retreat: int,
        train_gate: int,
        residual_gates: tuple[int, ...],
        frozen_residual_episode: Path | None,
        frozen_residual_gates: tuple[int, ...],
        frozen_action_episode: Path | None,
        frozen_action_gates: tuple[int, ...],
        actor_objective: str,
        awr_weight: float,
        awr_temperature: float,
        awr_max_weight: float,
        positive_td_priority: float,
        n_step: int,
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

        self.actor = GaussianActor(OBS_DIM, ACT_DIM).to(self.device)
        if is_residual:
            self.actor.load_state_dict(payload["actor"])
        else:
            # GaussianActor initializes to a zero-mean residual. Give it
            # useful but tightly bounded initial stochastic exploration.
            with torch.no_grad():
                self.actor.log_std.bias.fill_(-2.0)
        # Keep live action inference on CPU.  The MLP takes sub-millisecond
        # time there and cannot be starved by the simulator/GateNet sharing
        # the GPU near a gate.
        self.inference_actor = copy.deepcopy(self.actor).cpu().eval()
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
        self.return_scale = float(payload["return_scale"])
        self.gamma = float(payload["gamma"])
        self.alpha = float(alpha)
        self.bc_weight = float(bc_weight)
        self.sac_actor_weight = float(sac_actor_weight)
        self.trust_weight = float(trust_weight)
        self.teacher_blend = float(teacher_blend)
        self.residual_scale = float(residual_scale)
        self.residual_output_clip = float(np.clip(
            residual_output_clip,
            0.0,
            1.0,
        ))
        self.actor_warmup_updates = int(actor_warmup_updates)
        self.actor_lr = float(actor_lr)
        self.warmup_actor_lr = float(warmup_actor_lr)
        self.lateral_position_gain = float(lateral_position_gain)
        self.lateral_velocity_gain = float(lateral_velocity_gain)
        self.special_lateral_gate = int(special_lateral_gate)
        self.special_lateral_gain_scale = max(
            float(special_lateral_gain_scale), 0.0
        )
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
        self.trajectory_blend = float(np.clip(trajectory_blend, 0.0, 1.0))
        self.trajectory_controller = RateController(thrust_limit=0.52)
        self.reference_mode = str(reference_mode)
        self.reference_feedback_scale = max(
            float(reference_feedback_scale), 0.0
        )
        self.reference_thrust_scale = max(
            float(reference_thrust_scale), 0.0
        )
        self.reference_action_lead = int(reference_action_lead)
        self.gate4_action_lead = int(gate4_action_lead)
        self.reference_max_advance = max(1, int(reference_max_advance))
        self.reference_max_retreat = max(0, int(reference_max_retreat))
        self.train_gate = int(train_gate)
        self.residual_gates = frozenset(
            int(gate) for gate in residual_gates
            if 0 <= int(gate) < N_RACE_GATES
        )
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
        self.n_step = max(1, int(n_step))
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=actor_lr, weight_decay=1e-6
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=critic_lr, weight_decay=1e-6
        )
        self.demo = ReplayMemory(100_000, OBS_DIM)
        demo_npz = np.load(demo_path, allow_pickle=False)
        self.demo_observation = np.asarray(
            demo_npz["observation"], np.float32
        )
        self.demo_action = np.asarray(demo_npz["action"], np.float32)
        self.demo_velocity = np.asarray(
            demo_npz["velocity"], np.float32
        )
        self.demo_position = np.asarray(
            demo_npz["position"], np.float32
        )
        self.demo_gate = np.asarray(
            demo_npz["gate_index"], np.int16
        )
        self.demo_rotation = np.asarray([
            observation_rotation(row) for row in self.demo_observation
        ], np.float32)
        self.demo_gate_vector_world = np.einsum(
            "nij,nj->ni",
            self.demo_rotation,
            self.demo_observation[:, :3],
        ) * 10.0
        self.demo_gate_position = np.zeros((17, 3), np.float32)
        for gate_index in range(17):
            mask = self.demo_gate == gate_index
            self.demo_gate_position[gate_index] = np.median(
                self.demo_position[mask]
                + self.demo_gate_vector_world[mask],
                axis=0,
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
            "discount": np.full(
                len(demo_npz["reward"]), self.gamma, np.float32
            ),
        })
        self.live = ReplayMemory(500_000, OBS_DIM)
        self.exploration_state = np.zeros(ACT_DIM, np.float32)
        self.reference_gate: int | None = None
        self.reference_cursor: int | None = None
        self.updates = int(payload.get("updates", 0)) if is_residual else 0
        if is_residual:
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
        return (
            np.asarray(observation, np.float32) - self.observation_mean
        ) / self.observation_std

    def begin_episode(self) -> None:
        self.exploration_state.fill(0.0)
        self.reference_gate = None
        self.reference_cursor = None
        self.trajectory_controller.zi = 0.0

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

    @torch.inference_mode()
    def action(
        self,
        observation: np.ndarray,
        *,
        deterministic: bool,
        exploration_clip: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensor = torch.from_numpy(
            self.normalize(observation)
        ).unsqueeze(0)
        residual_mean = self.inference_actor.deterministic(tensor)
        teacher_mean = self.teacher.deterministic(tensor)
        gate_index = int(np.argmax(observation[34:51]))
        candidates = np.flatnonzero(self.demo_gate == gate_index)
        if not len(candidates):
            raise RuntimeError(
                f"clean demonstration has no samples for gate {gate_index}"
            )
        if self.reference_gate != gate_index:
            self.reference_gate = gate_index
            self.reference_cursor = int(candidates[0])
        cursor = int(self.reference_cursor)
        current_rotation = observation_rotation(observation)
        current_gate_vector_world = (
            current_rotation @ np.asarray(observation[:3]) * 10.0
        )
        if self.reference_mode == "sequential":
            nearest = np.asarray([cursor], dtype=int)
            weights = np.ones(1, dtype=float)
            gate_end = int(candidates[-1])
            self.reference_cursor = min(cursor + 1, gate_end)
        else:
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
            gate_vector_error = (
                self.demo_gate_vector_world[candidates]
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
        action_lead = self.reference_action_lead
        if gate_index == 4:
            action_lead += self.gate4_action_lead
        action_rows = np.clip(
            nearest + action_lead,
            0,
            len(self.demo_action) - 1,
        )
        reference = torch.from_numpy(np.sum(
            self.demo_action[action_rows] * weights[:, None],
            axis=0,
        )).unsqueeze(0)
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
        position_error_world = (
            reference_gate_vector_world - current_gate_vector_world
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
        tracking = np.zeros(ACT_DIM, np.float32)
        tracking[:3] = np.clip(
            np.asarray(attitude_error)
            * np.array([0.70, 0.70, 0.60]),
            -0.25,
            0.25,
        )
        lateral_gain_scale = (
            self.special_lateral_gain_scale
            if gate_index == self.special_lateral_gate
            else 1.0
        )
        tracking[0] += np.clip(
            -lateral_gain_scale
            * self.lateral_position_gain
            * position_error_reference[1]
            - lateral_gain_scale
            * self.lateral_velocity_gain
            * velocity_error_reference[1],
            -0.20,
            0.20,
        )
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
        vertical_gain_scale = (
            self.special_vertical_gain_scale
            if gate_index == self.special_vertical_gate
            else 1.0
        )
        tracking[3] = np.clip(
            vertical_gain_scale
            * self.vertical_position_gain
            * position_error_world[2]
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
        reference = reference + self.reference_feedback_scale * (
            torch.from_numpy(tracking).unsqueeze(0)
        )
        if self.reference_thrust_scale != 1.0:
            # Scale only thrust above/below hover rather than multiplying
            # collective outright. This slows the racing line without
            # starving the vehicle of gravity compensation.
            reference_thrust = 0.5 * (reference[:, 3] + 1.0)
            reference_thrust = (
                0.25
                + self.reference_thrust_scale
                * (reference_thrust - 0.25)
            )
            reference[:, 3] = torch.clamp(
                2.0 * reference_thrust - 1.0,
                -1.0,
                1.0,
            )
        if self.trajectory_blend > 0.0:
            current_position_world = (
                self.demo_gate_position[gate_index]
                - current_gate_vector_world
            )
            reference_position_world = np.sum(
                self.demo_position[nearest] * weights[:, None],
                axis=0,
            )
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
                    reference_velocity_world,
                    yaw_des,
                    1.0 / 30.0,
                )
            )
            trajectory_action = np.empty(ACT_DIM, np.float32)
            trajectory_action[:3] = (
                RATE_CMD_SIGN * RATE_CMD_GAIN * desired_rates
            ) / WIRE_RATE_LIMIT
            trajectory_action[3] = 2.0 * desired_thrust - 1.0
            trajectory_action = np.clip(
                trajectory_action, -1.0, 1.0
            )
            reference = (
                (1.0 - self.trajectory_blend) * reference
                + self.trajectory_blend
                * torch.from_numpy(trajectory_action).unsqueeze(0)
            )
        base = (
            self.teacher_blend * reference
            + (1.0 - self.teacher_blend) * teacher_mean
        )
        residual_active = self._residual_gate_active(gate_index)
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
        frozen_action = self._frozen_action(
            gate_index, int(self.reference_cursor)
        )
        if frozen_action is not None and gate_index != self.train_gate:
            selected = torch.from_numpy(
                np.clip(frozen_action, -1.0, 1.0)
            ).unsqueeze(0)
        else:
            selected = torch.clamp(
                base + self.residual_scale * residual,
                -1.0,
                1.0,
            )
        return (
            selected[0].cpu().numpy().astype(np.float32),
            residual[0].cpu().numpy().astype(np.float32),
            residual_mean[0].cpu().numpy().astype(np.float32),
        )

    def add_episode(self, transitions: list[dict]) -> None:
        """Add n-step rows and promote every passed-gate approach for SIL."""
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
                * (1.0 + abs(float(reward))),
                is_demo=row in successful_rows,
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
        return {
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
            "residual_scale": self.residual_scale,
            "residual_output_clip": self.residual_output_clip,
            "actor_warmup_updates": self.actor_warmup_updates,
            "actor_lr": self.actor_lr,
            "warmup_actor_lr": self.warmup_actor_lr,
            "lateral_position_gain": self.lateral_position_gain,
            "lateral_velocity_gain": self.lateral_velocity_gain,
            "lateral_bias_gates": sorted(self.lateral_bias_gates),
            "lateral_action_bias": self.lateral_action_bias,
            "right_lateral_bias_gates": sorted(
                self.right_lateral_bias_gates
            ),
            "right_lateral_action_bias": self.right_lateral_action_bias,
            "reference_action_lead": self.reference_action_lead,
            "reference_mode": self.reference_mode,
            "reference_feedback_scale": self.reference_feedback_scale,
            "reference_thrust_scale": self.reference_thrust_scale,
            "vertical_bias_gates": sorted(self.vertical_bias_gates),
            "vertical_action_bias": self.vertical_action_bias,
            "trajectory_blend": self.trajectory_blend,
            "gate4_action_lead": self.gate4_action_lead,
            "reference_max_advance": self.reference_max_advance,
            "reference_max_retreat": self.reference_max_retreat,
            "train_gate": self.train_gate,
            "residual_gates": sorted(self.residual_gates),
            "frozen_residual_gates": sorted(
                self.frozen_residual_profiles
            ),
            "frozen_action_gates": sorted(self.frozen_action_profiles),
            "actor_objective": self.actor_objective,
            "awr_weight": self.awr_weight,
            "awr_temperature": self.awr_temperature,
            "awr_max_weight": self.awr_max_weight,
            "positive_td_priority": self.positive_td_priority,
            "n_step": self.n_step,
            "updates": self.updates,
            "live_replay_size": self.live.size,
        }

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
    parser.add_argument("--episodes", type=int, default=50)
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
    parser.add_argument("--exploration-clip", type=float, default=0.08)
    parser.add_argument("--residual-scale", type=float, default=0.08)
    parser.add_argument(
        "--residual-output-clip",
        type=float,
        default=1.0,
        help=(
            "Hard clip applied to normalized residual actions in collection, "
            "evaluation, and Bellman targets."
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
        "--trajectory-blend",
        type=float,
        default=0.0,
        help=(
            "Blend a full 3-D position/velocity tracker around the clean "
            "demonstration into the demonstrated action."
        ),
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
        "--direct-position-pins",
        action="store_true",
        help="Strongly pin EKF position from complete fused active-gate quads.",
    )
    parser.add_argument("--reference-action-lead", type=int, default=0)
    parser.add_argument("--gate4-action-lead", type=int, default=0)
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
        "--seed-checkpoint",
        type=Path,
        default=REPO / "data/models/vq2_bc_seed.pt",
    )
    parser.add_argument(
        "--demo",
        type=Path,
        default=REPO / "data/vq2_sac_clean_demo.npz",
    )
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
        default=REPO / "data/vq2_sac_runs",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    single_instance_guard = acquire_single_instance_guard()
    affinity = set_process_cpu_affinity(args.cpu_affinity)
    if affinity is not None:
        print(
            f"Trainer CPU affinity locked to 0x{affinity:X}",
            flush=True,
        )
    torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.set_num_interop_threads(max(1, int(args.torch_interop_threads)))
    np.random.seed(11)
    torch.manual_seed(11)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    learner = VQ2SACLearner(
        args.seed_checkpoint,
        args.demo,
        device=args.device,
        actor_lr=args.actor_lr,
        warmup_actor_lr=args.warmup_actor_lr,
        critic_lr=args.critic_lr,
        alpha=args.alpha,
        bc_weight=args.bc_weight,
        sac_actor_weight=args.sac_actor_weight,
        trust_weight=args.trust_weight,
        teacher_blend=args.teacher_blend,
        residual_scale=args.residual_scale,
        residual_output_clip=args.residual_output_clip,
        actor_warmup_updates=args.actor_warmup_updates,
        lateral_position_gain=args.lateral_position_gain,
        lateral_velocity_gain=args.lateral_velocity_gain,
        special_lateral_gate=args.special_lateral_gate,
        special_lateral_gain_scale=args.special_lateral_gain_scale,
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
        trajectory_blend=args.trajectory_blend,
        reference_mode=args.reference_mode,
        reference_feedback_scale=args.reference_feedback_scale,
        reference_thrust_scale=args.reference_thrust_scale,
        reference_action_lead=args.reference_action_lead,
        gate4_action_lead=args.gate4_action_lead,
        reference_max_advance=args.reference_max_advance,
        reference_max_retreat=args.reference_max_retreat,
        train_gate=args.train_gate,
        residual_gates=tuple(
            int(value.strip())
            for value in args.residual_gates.split(",")
            if value.strip()
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
        n_step=args.n_step,
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
    elif args.replay_dir is not None:
        replay_episodes, replay_rows = learner.load_episode_directory(
            args.replay_dir
        )
        print(
            f"reconstructed {replay_rows} rows from {replay_episodes} "
            f"timing-healthy episodes in {args.replay_dir}",
            flush=True,
        )
    config = VQ2EnvConfig(
        control_hz=args.control_hz,
        thrust_cap=args.thrust_cap,
        launch_assist_s=args.launch_assist,
        launch_min_thrust=args.launch_min_thrust,
        speed_cap_mps=args.speed_cap,
        max_tilt_deg=args.max_tilt,
        anchor_timeout_s=args.anchor_timeout,
        gate_timeout_s=args.gate_timeout,
        max_camera_packet_age_s=args.max_camera_age,
        max_imu_packet_age_s=args.max_imu_age,
        max_transition_sim_s=args.max_transition_sim_time,
    )
    (run_dir / "config.json").write_text(json.dumps({
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": asdict(config),
    }, indent=2))

    mavlink = None
    vision = None
    environment = None
    dashboard = None
    stopping = False

    def stop_handler(_signal=None, _frame=None):
        nonlocal stopping
        stopping = True
        if environment is not None:
            environment.shutdown_to_spawn()

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    try:
        mavlink = MavIO(port=args.mav_port)
        vision = VisionRX(port=args.camera_port)
        localizer = LiveVQ2Localizer(
            mavlink=mavlink,
            vision=vision,
            map_path=args.map,
            primary_checkpoint=args.primary,
            refine_checkpoint=args.refiner,
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

        for episode in range(args.episodes):
            if stopping:
                break
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
            learner.begin_episode()
            transitions = []
            episode_reward = 0.0
            started = time.time()
            print(
                f"episode {episode:04d} START deterministic={deterministic} "
                f"anchor={reset_info['anchor']}",
                flush=True,
            )
            final_info = {}
            while not stopping:
                action, residual_action, actor_mean = learner.action(
                    observation,
                    deterministic=deterministic,
                    exploration_clip=exploration_clip,
                )
                next_observation, reward, terminated, truncated, info = (
                    environment.step(action)
                )
                transitions.append({
                    "observation": observation.copy(),
                    "action": residual_action.copy(),
                    "wire_action": np.asarray(
                        info["action"], np.float32
                    ),
                    "actor_mean": actor_mean.copy(),
                    "reference_row": int(learner.reference_cursor),
                    "reward": reward,
                    "next_observation": next_observation.copy(),
                    "done": float(terminated),
                    # Label the state that selected the action, not the
                    # post-step target (which advances on a crossing).
                    "gate_index": int(np.argmax(observation[34:51])),
                    "timing_healthy": bool(
                        info.get("timing_healthy", True)
                    ),
                    "camera_age_s": float(info["camera_age_s"]),
                    "imu_age_s": float(info["imu_age_s"]),
                    "sim_step_s": float(info["sim_step_s"]),
                    "step_ms": float(info["step_ms"]),
                })
                step_payload = {
                    "episode": episode,
                    "step": len(transitions) - 1,
                    "reward": reward,
                    **info,
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
                        "velocity": environment.localizer.state(
                        ).velocity.tolist(),
                    })
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
                if terminated or truncated:
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
            timing_healthy = bool(
                transitions
                and all(row["timing_healthy"] for row in transitions)
            )
            if timing_healthy:
                timing_failure_streak = 0
            else:
                timing_failure_streak += 1
            summary = {
                "episode": episode,
                "steps": len(transitions),
                "duration_s": duration,
                "reward": episode_reward,
                "gate_reached": reached,
                "finished": finished,
                "failure": final_info.get("failure"),
                "deterministic": deterministic,
                "evaluation": scheduled_evaluation,
                "exploration_clip": exploration_clip,
                "live_replay_size": learner.live.size,
                "updates": learner.updates,
                "timing_healthy": timing_healthy,
                "timing_failure_streak": timing_failure_streak,
                "camera_age_p95_s": float(np.percentile(
                    camera_ages, 95
                )),
                "camera_age_max_s": float(np.max(camera_ages)),
                "imu_age_p95_s": float(np.percentile(imu_ages, 95)),
                "imu_age_max_s": float(np.max(imu_ages)),
                "sim_step_p95_s": float(np.percentile(sim_steps, 95)),
                "sim_step_max_s": float(np.max(sim_steps)),
                "step_p95_ms": float(np.percentile(step_times, 95)),
                "step_max_ms": float(np.max(step_times)),
            }
            write_jsonl(run_dir / "episodes.jsonl", summary)
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
                actor_mean=np.asarray([
                    row["actor_mean"] for row in transitions
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
                    else:
                        # Any evaluation that does not clear the curriculum
                        # gate is a guardrail failure.  Previously, failures
                        # before the curriculum gate were ignored, allowing a
                        # regressed policy/runtime state to degrade for the
                        # remainder of a run without restoring best.pt.
                        early_failures += 1
                        print(
                            "EVAL: curriculum gate was not cleared; "
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
    print(f"run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
