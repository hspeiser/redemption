"""Vectorized replica of the live SAC trainer's demonstrated-line teacher.

Unlike the legacy fastsim RefController, this mirrors the actual nearest-row
metric, weighted feedforward lookup, frame-consistent feedback, per-gate lead,
thrust scaling, and biases used by ``train_vq2_sac_live.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter

from aigp.rl.vq2_features import WIRE_RATE_LIMIT

from aigp.fastsim.worldmodel import rotation_log


class LiveTeacherController:
    def __init__(
        self,
        demo_path: str | Path,
        n_envs: int,
        *,
        device: str = "cuda",
        schedule_gate_positions: np.ndarray | None = None,
        action_leads: np.ndarray | None = None,
        thrust_scales: np.ndarray | None = None,
        reference_rate_scales: np.ndarray | None = None,
        trajectory_velocity_scales: np.ndarray | None = None,
        trajectory_blends: np.ndarray | None = None,
        reference_gate_offsets_world: np.ndarray | None = None,
        predictive_handoff_distances: np.ndarray | None = None,
        reference_sequential_speeds: np.ndarray | None = None,
        reference_sequential_enabled: np.ndarray | None = None,
        rate_gain: np.ndarray | None = None,
        reference_max_advance: int = 8,
        reference_max_retreat: int = 2,
        lateral_position_gain: float = 0.08,
        lateral_velocity_gain: float = 0.04,
        longitudinal_position_gains: np.ndarray | None = None,
        longitudinal_velocity_gains: np.ndarray | None = None,
        lateral_feedback_limits: np.ndarray | None = None,
        vertical_position_gain: float = 0.30,
        vertical_velocity_gain: float = 0.10,
        vertical_gain_scales: np.ndarray | None = None,
        lateral_bias_gates: tuple[int, ...] = (),
        lateral_action_bias: float = 0.0,
        right_lateral_bias_gates: tuple[int, ...] = (),
        right_lateral_action_bias: float = 0.0,
        extra_lateral_biases: np.ndarray | None = None,
        vertical_bias_gates: tuple[int, ...] = (3, 5, 7, 8),
        vertical_action_bias: float = 0.05,
        lateral_gain_scales: np.ndarray | None = None,
        extra_vertical_biases: np.ndarray | None = None,
        gate_center_funnel_gates: tuple[int, ...] = (),
        gate_center_funnel_distance: float = 0.0,
        gate_center_funnel_full_distance: float = 0.0,
        gate_center_funnel_strength: float = 0.0,
        feedback_scale: float = 1.0,
    ) -> None:
        dev = torch.device(device)
        data = np.load(demo_path)
        tt = lambda value: torch.as_tensor(
            value, dtype=torch.float32, device=dev
        )
        self.obs = tt(data["observation"])
        self.action_table = tt(data["action"])
        self.position = tt(data["position"])
        self.velocity = tt(data["velocity"])
        self.n_pts = len(self.position)
        wall = np.asarray(data["wall"], np.float64)
        velocity_np = np.asarray(data["velocity"], np.float64)
        acceleration = np.zeros_like(velocity_np)
        source = (
            np.asarray(data["source_episode"]).astype(str)
            if "source_episode" in data.files else None
        )
        groups = (
            [np.arange(len(wall))]
            if source is None or source.ndim == 0
            else [np.flatnonzero(source == name)
                  for name in dict.fromkeys(source.tolist())]
        )
        for rows_np in groups:
            if len(rows_np) < 5:
                continue
            dt = float(np.median(np.diff(wall[rows_np])))
            if not np.isfinite(dt) or dt <= 1e-4:
                dt = 1.0 / 30.0
            window = min(15, len(rows_np) if len(rows_np) % 2 else len(rows_np) - 1)
            smooth = savgol_filter(
                velocity_np[rows_np], window, min(3, window - 2), axis=0,
                mode="interp",
            )
            acceleration[rows_np] = savgol_filter(
                smooth, window, min(3, window - 2), deriv=1, delta=dt,
                axis=0, mode="interp",
            )
        magnitude = np.linalg.norm(acceleration, axis=1)
        acceleration *= np.minimum(
            1.0, 18.0 / np.maximum(magnitude, 1e-6)
        )[:, None]
        self.acceleration = tt(acceleration)
        self.gate = torch.as_tensor(
            data["gate_index"], dtype=torch.long, device=dev
        )
        first = self.obs[:, 21:24]
        first = torch.nn.functional.normalize(first, dim=1)
        second = self.obs[:, 24:27]
        second = second - first * (first * second).sum(1, keepdim=True)
        second = torch.nn.functional.normalize(second, dim=1)
        third = torch.linalg.cross(first, second)
        self.rotation = torch.stack([first, second, third], dim=2)
        gate_vector_body = self.obs[:, :3] * 10.0
        self.gate_vector = torch.einsum(
            "nij,nj->ni", self.rotation, gate_vector_body
        )
        self.gate_position = torch.zeros(17, 3, device=dev)
        self.gate_normal = torch.zeros(17, 3, device=dev)
        self.gate_start = torch.zeros(17, dtype=torch.long, device=dev)
        self.gate_end = torch.zeros(17, dtype=torch.long, device=dev)
        for gate_index in range(17):
            rows = torch.nonzero(self.gate == gate_index).squeeze(1)
            self.gate_start[gate_index] = rows[0]
            self.gate_end[gate_index] = rows[-1]
            self.gate_position[gate_index] = torch.median(
                self.position[rows] + self.gate_vector[rows], dim=0
            ).values
            normal_world = torch.einsum(
                "nij,nj->ni", self.rotation[rows], self.obs[rows, 3:6]
            )
            self.gate_normal[gate_index] = torch.nn.functional.normalize(
                torch.median(normal_world, dim=0).values, dim=0
            )
        # The live controller receives gate-relative observations in the
        # runtime map frame, then translates the inferred drone position into
        # the demonstration frame before matching/tracking the reference.  A
        # surveyed runtime gate can differ from the gate center reconstructed
        # from the demo by decimetres, so treating these frames as identical
        # changes both row selection and feedback near the aperture.
        if schedule_gate_positions is None:
            self.schedule_gate_position = self.gate_position.clone()
        else:
            schedule = np.asarray(schedule_gate_positions, np.float32)
            if schedule.shape != (17, 3):
                raise ValueError(
                    "schedule_gate_positions must have shape (17, 3), "
                    f"got {schedule.shape}"
                )
            self.schedule_gate_position = tt(schedule)
        self.demo_speed_scaled = torch.linalg.norm(
            self.obs[:, 18:21], dim=1
        )
        if action_leads is None:
            action_leads = np.zeros((n_envs, 5), np.int64)
        if thrust_scales is None:
            thrust_scales = np.ones((n_envs, 5), np.float32)
        if reference_rate_scales is None:
            reference_rate_scales = np.ones(5, np.float32)
        if trajectory_velocity_scales is None:
            trajectory_velocity_scales = np.ones((n_envs, 5), np.float32)
        if trajectory_blends is None:
            trajectory_blends = np.zeros((n_envs, 5), np.float32)
        if predictive_handoff_distances is None:
            predictive_handoff_distances = np.zeros(
                (n_envs, 5), np.float32
            )
        if reference_sequential_speeds is None:
            reference_sequential_speeds = np.ones(5, np.float32)
        if reference_sequential_enabled is None:
            reference_sequential_enabled = np.zeros(5, bool)
        self.action_leads = torch.as_tensor(
            action_leads, dtype=torch.long, device=dev
        )
        self.parameter_gate_count = int(self.action_leads.shape[1])
        if not 1 <= self.parameter_gate_count <= 17:
            raise ValueError(
                "per-gate controller arrays must cover 1 through 17 gates, "
                f"got {self.parameter_gate_count}"
            )
        if reference_gate_offsets_world is None:
            reference_gate_offsets_world = np.zeros(
                (n_envs, self.parameter_gate_count, 3), np.float32
            )
        reference_offsets = np.asarray(
            reference_gate_offsets_world, np.float32
        )
        expected_offset_shape = (n_envs, self.parameter_gate_count, 3)
        if reference_offsets.shape != expected_offset_shape:
            raise ValueError(
                "reference_gate_offsets_world must have shape "
                f"{expected_offset_shape}, got {reference_offsets.shape}"
            )
        self.reference_gate_offsets_world = tt(reference_offsets)
        self.thrust_scales = tt(thrust_scales)
        rate_scales = np.asarray(reference_rate_scales, np.float32)
        if rate_scales.shape not in (
            (self.parameter_gate_count,),
            (n_envs, self.parameter_gate_count),
        ):
            raise ValueError(
                "reference_rate_scales must have shape "
                f"({self.parameter_gate_count},) or "
                f"({n_envs}, {self.parameter_gate_count}), got "
                f"{rate_scales.shape}"
            )
        self.reference_rate_scales = tt(rate_scales)
        self.trajectory_velocity_scales = tt(trajectory_velocity_scales)
        self.trajectory_blends = tt(trajectory_blends)
        self.predictive_handoff_distances = tt(
            predictive_handoff_distances
        )
        self.reference_sequential_speeds = tt(
            reference_sequential_speeds
        )
        self.reference_sequential_enabled = torch.as_tensor(
            reference_sequential_enabled, dtype=torch.bool, device=dev
        )
        gain = np.abs(
            np.asarray(rate_gain if rate_gain is not None
                       else [2.51, 2.62, 2.00], np.float32)
        )
        self.trajectory_rate_denominator = tt(gain * WIRE_RATE_LIMIT)
        self.trajectory_zi = torch.zeros(n_envs, device=dev)
        self.n_envs = n_envs
        self.dev = dev
        self.max_advance = int(reference_max_advance)
        self.max_retreat = int(reference_max_retreat)
        self.lp = float(lateral_position_gain)
        self.lv = float(lateral_velocity_gain)
        if longitudinal_position_gains is None:
            longitudinal_position_gains = np.zeros(5, np.float32)
        if longitudinal_velocity_gains is None:
            longitudinal_velocity_gains = np.zeros(5, np.float32)
        if lateral_feedback_limits is None:
            lateral_feedback_limits = np.full(5, 0.20, np.float32)
        if vertical_gain_scales is None:
            vertical_gain_scales = np.ones(5, np.float32)
        if extra_lateral_biases is None:
            extra_lateral_biases = np.zeros(5, np.float32)
        self.longitudinal_position_gains = tt(
            longitudinal_position_gains
        )
        self.longitudinal_velocity_gains = tt(
            longitudinal_velocity_gains
        )
        self.lateral_feedback_limits = tt(lateral_feedback_limits)
        self.vertical_gain_scales = tt(vertical_gain_scales)
        self.lateral_bias_gates = tuple(lateral_bias_gates)
        self.lateral_action_bias = float(lateral_action_bias)
        self.right_lateral_bias_gates = tuple(right_lateral_bias_gates)
        self.right_lateral_action_bias = float(right_lateral_action_bias)
        self.extra_lateral_biases = tt(extra_lateral_biases)
        self.vp = float(vertical_position_gain)
        self.vv = float(vertical_velocity_gain)
        self.vertical_bias_gates = tuple(vertical_bias_gates)
        self.vertical_action_bias = float(vertical_action_bias)
        if lateral_gain_scales is None:
            lateral_gain_scales = np.ones(5, np.float32)
        if extra_vertical_biases is None:
            extra_vertical_biases = np.zeros(5, np.float32)
        self.lateral_gain_scales = tt(lateral_gain_scales)
        self.extra_vertical_biases = tt(extra_vertical_biases)
        self.funnel_gates = set(int(g) for g in gate_center_funnel_gates)
        self.funnel_distance = float(gate_center_funnel_distance)
        self.funnel_full_distance = float(gate_center_funnel_full_distance)
        self.funnel_strength = float(gate_center_funnel_strength)
        self.feedback_scale = float(feedback_scale)
        self.target = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.reference_gate = torch.full(
            (n_envs,), -1, dtype=torch.long, device=dev
        )
        self.handoff_latched_gate = torch.full(
            (n_envs,), -1, dtype=torch.long, device=dev
        )
        self.idx = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.reference_cursor = torch.zeros(
            n_envs, dtype=torch.float32, device=dev
        )
        self.previous_action = torch.zeros(n_envs, 4, device=dev)
        self.last_reference_row = torch.zeros(
            n_envs, dtype=torch.long, device=dev
        )

    def set_target(self, target: torch.Tensor) -> None:
        self.target.copy_(torch.clamp(target, 0, 16))

    def set_previous_action(self, action: torch.Tensor) -> None:
        self.previous_action.copy_(action)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.reference_gate[env_ids] = -1
        self.handoff_latched_gate[env_ids] = -1
        self.idx[env_ids] = 0
        self.reference_cursor[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.trajectory_zi[env_ids] = 0.0

    def reset_nearest(self, env_ids: torch.Tensor, _position: torch.Tensor) -> None:
        self.reset(env_ids)

    @torch.no_grad()
    def action(self, p: torch.Tensor, v: torch.Tensor,
               rotation: torch.Tensor) -> torch.Tensor:
        n = len(p)
        row = torch.arange(n, device=self.dev)
        event_gate = self.target
        stale_latch = (
            (self.handoff_latched_gate >= 0)
            & (self.handoff_latched_gate != event_gate)
        )
        self.handoff_latched_gate = torch.where(
            stale_latch,
            torch.full_like(self.handoff_latched_gate, -1),
            self.handoff_latched_gate,
        )
        handoff_lookup = torch.clamp(
            event_gate, max=self.parameter_gate_count - 1
        )
        handoff_distance = self.predictive_handoff_distances[
            row, handoff_lookup
        ]
        event_gate_vector = self.schedule_gate_position[event_gate] - p
        distance_to_event_gate = torch.linalg.norm(event_gate_vector, dim=1)
        trigger_handoff = (
            (handoff_distance > 0.0)
            & (event_gate < 16)
            & (distance_to_event_gate <= handoff_distance)
        )
        self.handoff_latched_gate = torch.where(
            trigger_handoff,
            event_gate,
            self.handoff_latched_gate,
        )
        handoff_active = self.handoff_latched_gate == event_gate
        gate = torch.where(
            handoff_active,
            torch.clamp(event_gate + 1, max=16),
            event_gate,
        )
        # Exact live-frame transform.  The observation vector is anchored to
        # the runtime map's current event gate.  Re-anchor that same vector to
        # the demonstrated gate center, producing the position used by every
        # downstream reference calculation.  During predictive handoff this
        # position remains fixed while the control gate advances.
        controller_p = self.gate_position[event_gate] - event_gate_vector
        changed = self.reference_gate != gate
        self.idx = torch.where(changed, self.gate_start[gate], self.idx)
        self.reference_cursor = torch.where(
            changed,
            self.gate_start[gate].float(),
            self.reference_cursor,
        )
        self.reference_gate.copy_(gate)
        start = self.gate_start[gate]
        end = self.gate_end[gate]
        lookup_gate = torch.clamp(gate, max=self.parameter_gate_count - 1)
        sequential = self.reference_sequential_enabled[lookup_gate]
        sequential_speed = self.reference_sequential_speeds[lookup_gate]
        offsets = torch.arange(
            -self.max_retreat, self.max_advance + 1, device=self.dev
        )
        candidates = self.idx[:, None] + offsets[None]
        valid = (candidates >= start[:, None]) & (candidates <= end[:, None])
        candidates = torch.maximum(
            torch.minimum(candidates, end[:, None]), start[:, None]
        )
        current_gate_vector = self.gate_position[gate] - controller_p
        # Deform the demonstrated line continuously between gate crossing
        # offsets.  Segment g starts at the previous gate's offset and ends
        # at gate g's offset, so changing the target gate cannot introduce a
        # reference-position jump.  The runtime map itself is never moved.
        current_offset_valid = gate < self.parameter_gate_count
        current_offset = self.reference_gate_offsets_world[
            row, lookup_gate
        ] * current_offset_valid[:, None]
        previous_lookup = torch.clamp(
            gate - 1, min=0, max=self.parameter_gate_count - 1
        )
        previous_offset_valid = (
            (gate > 0) & ((gate - 1) < self.parameter_gate_count)
        )
        previous_offset = self.reference_gate_offsets_world[
            row, previous_lookup
        ] * previous_offset_valid[:, None]
        segment_rows = torch.clamp(end - start, min=1).float()
        candidate_phase = (
            (candidates - start[:, None]).float()
            / segment_rows[:, None]
        )
        candidate_offsets = (
            previous_offset[:, None]
            + candidate_phase[:, :, None]
            * (current_offset - previous_offset)[:, None]
        )
        candidate_gate_vector = (
            self.gate_vector[candidates] - candidate_offsets
        )
        gate_error = candidate_gate_vector - current_gate_vector[:, None]
        previous_error = (
            self.obs[candidates, 30:34] - self.previous_action[:, None]
        ).square().sum(2)
        speed_error = (
            self.demo_speed_scaled[candidates]
            - torch.linalg.norm(v, dim=1)[:, None] / 10.0
        ).square()
        distance = 4.0 * gate_error.square().sum(2) \
            + 2.0 * previous_error + 0.5 * speed_error
        distance = torch.where(
            valid, distance, torch.full_like(distance, 1e9)
        )
        best = distance.argmin(1)
        nearest_idx = candidates[row, best]
        top_distance, top_local = torch.topk(
            distance, k=4, dim=1, largest=False
        )
        nearest = candidates.gather(1, top_local)
        weights = 1.0 / torch.clamp(top_distance, min=1e-4)
        weights = weights / weights.sum(1, keepdim=True)
        # The live controller can advance selected gates by elapsed control
        # steps instead of re-projecting onto the line from a noisy EKF pose.
        # Preserve a fractional cursor so speeds such as 0.95/1.05 round-trip
        # exactly rather than accumulating integer truncation.
        cursor = torch.maximum(
            torch.minimum(self.reference_cursor, end.float()),
            start.float(),
        )
        lower = torch.floor(cursor).long()
        upper = torch.minimum(lower + 1, end)
        fraction = cursor - lower.float()
        sequential_nearest = torch.stack(
            [lower, upper, lower, upper], dim=1
        )
        sequential_weights = torch.stack(
            [1.0 - fraction, fraction,
             torch.zeros_like(fraction), torch.zeros_like(fraction)],
            dim=1,
        )
        nearest = torch.where(
            sequential[:, None], sequential_nearest, nearest
        )
        weights = torch.where(
            sequential[:, None], sequential_weights, weights
        )
        self.idx = torch.where(sequential, lower, nearest_idx)
        self.reference_cursor = torch.where(
            sequential,
            torch.minimum(cursor + sequential_speed, end.float()),
            nearest_idx.float(),
        )
        lead = self.action_leads[row, lookup_gate]
        action_rows = torch.maximum(
            torch.minimum(nearest + lead[:, None], end[:, None]),
            start[:, None],
        )
        reference = (
            self.action_table[action_rows] * weights[:, :, None]
        ).sum(1)
        rate_scale = (
            self.reference_rate_scales[lookup_gate]
            if self.reference_rate_scales.ndim == 1
            else self.reference_rate_scales[row, lookup_gate]
        )
        reference[:, :3] = torch.clamp(
            reference[:, :3] * rate_scale[:, None], -1.0, 1.0
        )
        # Weighted reference state. Rotation averaging is safe over four
        # adjacent 30 Hz rows; re-orthonormalize the first two columns.
        # The live controller reconstructs the reference position from its
        # robust per-gate center and the weighted gate-relative vector. It
        # does not directly average the recorded odometry positions. The two
        # differ by centimeters in the demonstration, enough to perturb the
        # blended trajectory controller at a gate handoff.
        nearest_phase = (
            (nearest - start[:, None]).float()
            / segment_rows[:, None]
        )
        nearest_offsets = (
            previous_offset[:, None]
            + nearest_phase[:, :, None]
            * (current_offset - previous_offset)[:, None]
        )
        ref_gate_vector = (
            (self.gate_vector[nearest] - nearest_offsets)
            * weights[:, :, None]
        ).sum(1)
        ref_position = self.gate_position[gate] - ref_gate_vector
        ref_velocity = (self.velocity[nearest] * weights[:, :, None]).sum(1)
        ref_acceleration = (
            self.acceleration[nearest] * weights[:, :, None]
        ).sum(1)
        ref_rotation_raw = (
            self.rotation[nearest] * weights[:, :, None, None]
        ).sum(1)
        x = torch.nn.functional.normalize(ref_rotation_raw[:, :, 0], dim=1)
        y = ref_rotation_raw[:, :, 1] - x * (
            x * ref_rotation_raw[:, :, 1]
        ).sum(1, keepdim=True)
        y = torch.nn.functional.normalize(y, dim=1)
        z = torch.linalg.cross(x, y)
        ref_rotation = torch.stack([x, y, z], dim=2)
        # Same gate-centering funnel used by the live teacher.  It only alters
        # the in-plane reference close to explicitly selected gates.
        if self.funnel_distance > 0.0 and self.funnel_gates:
            normal = self.gate_normal[gate]
            plane_distance = torch.abs(
                ((controller_p - self.gate_position[gate]) * normal).sum(1)
            )
            enabled = torch.zeros_like(plane_distance, dtype=torch.bool)
            for funnel_gate in self.funnel_gates:
                enabled |= gate == funnel_gate
            full = min(self.funnel_full_distance,
                       self.funnel_distance - 1e-3)
            weight = self.funnel_strength * torch.clamp(
                (self.funnel_distance - plane_distance)
                / max(self.funnel_distance - full, 1e-3), 0.0, 1.0
            )
            weight = weight * enabled.float()
            from_center = ref_position - self.gate_position[gate]
            in_plane = from_center - normal * (
                from_center * normal
            ).sum(1, keepdim=True)
            ref_position = ref_position - weight[:, None] * in_plane
        position_error_world = controller_p - ref_position
        velocity_error_world = v - ref_velocity
        attitude_error = rotation_log(
            rotation.transpose(1, 2) @ ref_rotation
        )
        position_error_ref = torch.einsum(
            "nij,nj->ni", ref_rotation.transpose(1, 2), position_error_world
        )
        velocity_error_ref = torch.einsum(
            "nij,nj->ni", ref_rotation.transpose(1, 2), velocity_error_world
        )
        sequential_velocity_error_ref = torch.einsum(
            "nij,nj->ni",
            ref_rotation.transpose(1, 2),
            v - sequential_speed[:, None] * ref_velocity,
        )
        tracking = torch.zeros_like(reference)
        tracking[:, :3] = torch.clamp(
            attitude_error * torch.tensor(
                [0.70, 0.70, 0.60], device=self.dev
            ), -0.25, 0.25,
        )
        gate5 = torch.clamp(gate, max=self.parameter_gate_count - 1)
        longitudinal = (
            self.longitudinal_position_gains[gate5]
            * position_error_ref[:, 0]
            + self.longitudinal_velocity_gains[gate5]
            * sequential_velocity_error_ref[:, 0]
        )
        tracking[:, 1] += torch.clamp(longitudinal, -0.20, 0.20)
        gain_scale = self.lateral_gain_scales[gate5]
        lateral = -gain_scale * self.lp * position_error_ref[:, 1] \
            - gain_scale * self.lv * velocity_error_ref[:, 1]
        limit = self.lateral_feedback_limits[gate5]
        tracking[:, 0] += torch.maximum(
            torch.minimum(lateral, limit), -limit
        )
        for bias_gate in self.lateral_bias_gates:
            tracking[:, 0] += (gate == bias_gate).float() \
                * self.lateral_action_bias
        for bias_gate in self.right_lateral_bias_gates:
            tracking[:, 0] += (gate == bias_gate).float() \
                * self.right_lateral_action_bias
        tracking[:, 0] += self.extra_lateral_biases[gate5]
        tracking[:, 0] = torch.clamp(tracking[:, 0], -0.25, 0.25)
        vertical_scale = self.vertical_gain_scales[gate5]
        tracking[:, 3] = torch.clamp(
            vertical_scale * self.vp * position_error_world[:, 2]
            + vertical_scale * self.vv * velocity_error_world[:, 2],
            -0.15, 0.15,
        )
        for bias_gate in self.vertical_bias_gates:
            tracking[:, 3] += (gate == bias_gate).float() \
                * self.vertical_action_bias
            tracking[:, 3] = torch.clamp(tracking[:, 3], -0.15, 0.15)
        tracking[:, 3] += self.extra_vertical_biases[gate5]
        tracking[:, 3] = torch.clamp(tracking[:, 3], -0.15, 0.15)
        reference = reference + self.feedback_scale * tracking
        # Live applies its hover-centered thrust scaling after feedback has
        # been added. Keeping this order matters whenever a segment has both
        # vertical tracking error and a non-unit thrust scale (gate 0 here).
        scale = self.thrust_scales[row, lookup_gate]
        wire = 0.5 * (reference[:, 3] + 1.0)
        wire = 0.25 + scale * (wire - 0.25)
        reference[:, 3] = torch.clamp(2.0 * wire - 1.0, -1.0, 1.0)

        blend = self.trajectory_blends[row, lookup_gate]
        speed_scale = self.trajectory_velocity_scales[row, lookup_gate]
        if bool((blend > 0.0).any()):
            desired_velocity = speed_scale[:, None] * ref_velocity
            desired_acceleration = speed_scale[:, None].square() * ref_acceleration
            a_cmd = (
                torch.tensor([1.2, 1.2, 2.0], device=self.dev)
                * (ref_position - controller_p)
                + torch.tensor([2.0, 2.0, 2.8], device=self.dev)
                * (desired_velocity - v)
                + desired_acceleration
            )
            self.trajectory_zi = torch.clamp(
                self.trajectory_zi
                + 0.6 * (ref_position[:, 2] - controller_p[:, 2]) / 30.0,
                -3.0, 3.0,
            )
            a_cmd[:, 2] += self.trajectory_zi
            horizontal = torch.linalg.norm(a_cmd[:, :2], dim=1)
            hscale = torch.clamp(14.0 / torch.clamp(horizontal, min=1e-6), max=1.0)
            a_cmd[:, :2] *= hscale[:, None]
            a_cmd[:, 2] = torch.clamp(a_cmd[:, 2], -18.0, 12.0)
            t_des = a_cmd - torch.tensor([0.0, 0.0, 9.81], device=self.dev)
            t_des[:, 2] = torch.minimum(
                t_des[:, 2], torch.full_like(t_des[:, 2], -2.0)
            )
            z_des = -torch.nn.functional.normalize(t_des, dim=1)
            yaw = torch.atan2(ref_rotation[:, 1, 0], ref_rotation[:, 0, 0])
            x_course = torch.stack([
                torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)
            ], dim=1)
            y_des = torch.linalg.cross(z_des, x_course)
            y_des = torch.nn.functional.normalize(y_des, dim=1)
            x_des = torch.linalg.cross(y_des, z_des)
            desired_rotation = torch.stack([x_des, y_des, z_des], dim=2)
            desired_rates = 4.0 * rotation_log(
                rotation.transpose(1, 2) @ desired_rotation
            )
            desired_rates = torch.clamp(desired_rates, -4.0, 4.0)
            trajectory = torch.empty_like(reference)
            trajectory[:, :3] = torch.clamp(
                desired_rates / self.trajectory_rate_denominator, -1.0, 1.0
            )
            desired_thrust = torch.clamp(
                torch.linalg.norm(t_des, dim=1) / 40.0, 0.02, 0.52
            )
            trajectory[:, 3] = 2.0 * desired_thrust - 1.0
            reference = (
                (1.0 - blend[:, None]) * reference
                + blend[:, None] * trajectory
            )
        selected = torch.clamp(reference, -1.0, 1.0)
        self.previous_action.copy_(selected)
        self.last_reference_row.copy_(self.idx)
        return selected
