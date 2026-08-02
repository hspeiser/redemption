"""Live VQ2 SAC environment driven only by deployable vision and IMU.

Every clearly failed flight is a *terminal* transition.  The simulator can
only hard-reset to spawn, so allowing a missed-gate or lost-localization
episode to wander wastes both wall time and replay quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from aigp.mavlink_io import MavIO
from aigp.rl.vq2_features import (
    ACT_DIM,
    N_RACE_GATES,
    action_to_wire_command,
    build_observation,
    course_progress,
    load_gate_geometry,
)
from aigp.vq2_live_localizer import LiveVQ2Localizer


@dataclass(frozen=True)
class VQ2EnvConfig:
    control_hz: float = 30.0
    gate_bonus: float = 25.0
    finish_bonus: float = 600.0
    progress_scale: float = 2.0
    time_penalty_per_s: float = 0.8
    collision_penalty: float = 100.0
    failed_run_penalty: float = 60.0
    action_smoothness: float = 0.02
    thrust_cap: float = 0.52
    # The demonstrated manual lap ramps from nearly zero thrust. If one early
    # control interval is stretched by simulator jitter, that command can hold
    # the drone against the pad and create a reset loop. Guarantee a modest
    # launch thrust until the vehicle has had time to separate from the pad.
    launch_assist_s: float = 0.55
    launch_min_thrust: float = 0.30
    speed_cap_mps: float = 12.0
    # A single noisy EKF velocity sample must not kill an otherwise healthy
    # lap. Require a sustained excursion, and only re-arm the detector after
    # speed falls comfortably below the cap.
    overspeed_confirmation_s: float = 0.20
    overspeed_hysteresis_mps: float = 0.50
    # Hold the vehicle motionless until the simulator's authoritative race
    # clock reaches race_start_ms. This preserves the useful spawn anchor
    # work without risking an official early-start DQ.
    wait_for_official_start: bool = True
    race_start_timeout_s: float = 15.0
    # Exactly -1 keeps the historical packet-authoritative release. Any other
    # value projects the simulator boot clock forward from the newest pending
    # RACE_STATUS packet and releases at race_start_ms + margin. Positive
    # values start after the clock; negative values intentionally test an
    # early release (which may be disqualified by the simulator).
    official_release_margin_ms: float = -1.0
    anchor_timeout_s: float = 0.75
    # Hard-reset if the official gate index has not advanced for this much
    # simulator time.  This clock starts at race release and restarts only
    # when MAVLink confirms that the next gate was passed.
    gate_timeout_s: float = 4.5
    far_no_progress_s: float = 1.5
    near_no_progress_s: float = 3.5
    # VQ2 has long inter-gate views where a distant gate supplies no usable
    # corners for ~2 s even though camera and IMU packets remain healthy.
    # The measured IMU dead-reckoning is still recoverable there; kill only
    # after a genuinely multi-second landmark drought.
    max_visual_age_s: float = 2.8
    max_position_sigma_m: float = 2.0
    localization_grace_s: float = 0.30
    # These are packet freshness limits, not landmark-age limits.  A distant
    # gate can legitimately leave visual_age_s high while fresh camera frames
    # and IMU packets continue to arrive.
    max_camera_packet_age_s: float = 0.20
    max_imu_packet_age_s: float = 0.10
    # Tolerate isolated simulator/render hitches rather than reset-looping on
    # the pad. The actual elapsed simulator time is still logged per row.
    max_transition_sim_s: float = 0.30
    max_off_track_m: float = 6.0
    max_episode_s: float = 50.0
    miss_confirmation_s: float = 0.22
    # The official gate-pass event is a deployable one-dimensional landmark:
    # at receipt, the drone has just crossed the physical gate plane.  Keep
    # this opt-in until live A/B validates the correction.
    gate_event_plane_correction: bool = False
    gate_event_forward_offset_m: float = 0.0
    # Healthy expert/live laps stay below 47 degrees.  Terminate well before
    # the simulator's upside-down auto-respawn can silently teleport the
    # vehicle and let one episode continue across two physical attempts.
    max_tilt_deg: float = 80.0


def _race_pending(status: dict | None) -> bool:
    return bool(
        status
        and status["active_gate"] == 0
        and (
            status["race_start_ms"] < 0
            or status["sim_boot_ms"] < status["race_start_ms"]
        )
    )


def _race_active(status: dict | None) -> bool:
    return bool(
        status
        and status["race_start_ms"] >= 0
        and status["sim_boot_ms"] >= status["race_start_ms"]
        and status["race_finish_ns"] < 0
    )


def _distance_to_polyline(point: np.ndarray, points: np.ndarray) -> float:
    starts = points[:-1]
    segments = points[1:] - starts
    denominator = np.sum(segments * segments, axis=1)
    fraction = np.sum((point - starts) * segments, axis=1) / np.maximum(
        denominator, 1e-9
    )
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = starts + fraction[:, None] * segments
    return float(np.min(np.linalg.norm(closest - point, axis=1)))


class VQ2LiveEnv:
    """Small Gym-like API used by the live SAC trainer."""

    def __init__(
        self,
        mavlink: MavIO,
        localizer: LiveVQ2Localizer,
        config: VQ2EnvConfig | None = None,
    ) -> None:
        self.mavlink = mavlink
        self.localizer = localizer
        self.config = config or VQ2EnvConfig()
        self.gate_event_plane_correction_enabled = bool(
            self.config.gate_event_plane_correction
        )
        self.dt = 1.0 / self.config.control_hz
        self.previous_action = np.zeros(ACT_DIM, np.float32)
        self.geometry = None
        self.spawn_position = np.zeros(3)
        self.gate_normals = np.zeros((N_RACE_GATES, 3))
        self.track_points = np.zeros((N_RACE_GATES + 1, 3))
        self.target = 0
        self.episode_started_wall = 0.0
        self.last_step_wall = 0.0
        self.episode_started_sim = 0.0
        self.gate_started_sim = 0.0
        self.best_progress_sim = 0.0
        self.last_sim_us = 0
        self.next_sim_us = 0
        self.gate_started_wall = 0.0
        self.last_arm_wall = 0.0
        self.collision_after_wall_ns = 0
        self.previous_progress = 0.0
        self.episode_return = 0.0
        self.best_progress = 0.0
        self.best_progress_wall = 0.0
        self.previous_signed_plane: float | None = None
        self.pending_miss_sim: float | None = None
        self.overspeed_started_sim: float | None = None
        self.localization_bad_since: float | None = None
        self.last_status_boot_ms: int | None = None
        self.race_was_active = False
        self.predictive_release_pending = False
        self.steps = 0

    def emergency_stop(self) -> None:
        for _ in range(3):
            self.mavlink.send_attitude_rates(0.0, 0.0, 0.0, 0.0)
            self.mavlink.arm(False)
            time.sleep(0.01)

    def _request_immediate_respawn(self) -> None:
        """Reset the drone before removing thrust from an airborne terminal.

        A non-collision terminal can occur while the vehicle is flying.
        Disarming first makes it visibly fall while the asynchronous vision
        worker shuts down. The simulator reset command is the authoritative,
        recoverable way to return the drone to the pad.
        """
        for _ in range(3):
            self.mavlink.reset_sim()
            time.sleep(0.005)
        self.mavlink.arm(False)

    def shutdown_to_spawn(self) -> None:
        """Leave the simulator running, but return the drone safely to spawn."""
        self.localizer.stop_async()
        self._request_immediate_respawn()

    def park_after_episode(self) -> None:
        """Immediately return an ended flight to the spawn pad.

        Offline updates can take several seconds. Leaving an airborne drone
        disarmed for that interval looks like an unexplained fall and can
        generate a collision storm. Stop delayed vision and perform the
        normal in-simulator drone reset before learning begins.
        """
        self.localizer.stop_async()
        self.emergency_stop()
        self._hard_reset_to_countdown()

    def _keepalive(self) -> None:
        now = time.time()
        # VQ2 accepts controls during countdown.  Never apply hover thrust
        # while the drone is still resting at its -18 deg pad attitude.
        self.mavlink.send_attitude_rates(0.0, 0.0, 0.0, 0.0)
        if now - self.last_arm_wall >= 0.35:
            self.mavlink.arm(True)
            self.last_arm_wall = now

    def _hard_reset_to_countdown(self) -> None:
        deadline = time.time() + 25.0
        while time.time() < deadline:
            reset_sent_wall_ns = time.time_ns()
            for _ in range(6):
                self.mavlink.arm(False)
                self.mavlink.reset_sim()
                time.sleep(0.015)
            wait_deadline = time.time() + 1.5
            while time.time() < wait_deadline:
                status = self.mavlink.race_status
                if (
                    _race_pending(status)
                    and status["wall_ns"] > reset_sent_wall_ns
                    and status["sim_boot_ms"] < 1_500
                ):
                    # Let the reset impulse/UDP reorder tail clear before the
                    # anchor starts selecting stationary IMU samples.
                    time.sleep(0.20)
                    return
                time.sleep(0.02)
        raise RuntimeError("simulator did not enter VQ2 countdown after reset")

    def _configure_anchored_course(self) -> None:
        self.geometry = load_gate_geometry(self.localizer.gates)
        self.gate_normals = np.zeros((N_RACE_GATES, 3), float)
        previous = self.spawn_position
        for index, gate in enumerate(self.localizer.gates[:N_RACE_GATES]):
            qw, qx, qy, qz = gate["quat_wxyz"]
            rotation = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            normal = rotation[:, 1]
            incoming = np.asarray(gate["pos"], float) - previous
            if np.dot(normal, incoming) < 0.0:
                normal = -normal
            self.gate_normals[index] = normal / (
                np.linalg.norm(normal) + 1e-9
            )
            previous = np.asarray(gate["pos"], float)
        self.track_points = np.vstack([
            self.spawn_position,
            self.geometry.positions,
        ])

    def _wait_for_official_release(self) -> tuple[dict, float]:
        """Hold zero thrust until the simulator says the race is active."""
        started = time.time()
        deadline = started + self.config.race_start_timeout_s
        while time.time() < deadline:
            status = self.mavlink.race_status
            if _race_active(status):
                return status, time.time() - started
            if not _race_pending(status):
                raise RuntimeError(
                    f"invalid VQ2 race status during countdown: {status}"
                )
            margin_ms = float(self.config.official_release_margin_ms)
            if margin_ms != -1.0 and int(status["race_start_ms"]) >= 0:
                packet_age_ms = max(
                    0.0, (time.time_ns() - int(status["wall_ns"])) * 1e-6
                )
                projected_boot_ms = (
                    float(status["sim_boot_ms"]) + packet_age_ms
                )
                if projected_boot_ms >= (
                    float(status["race_start_ms"]) + margin_ms
                ):
                    projected = dict(status)
                    projected["source_sim_boot_ms"] = int(
                        status["sim_boot_ms"]
                    )
                    projected["sim_boot_ms"] = int(projected_boot_ms)
                    projected["predictive_release"] = True
                    return projected, time.time() - started
            self.mavlink.send_attitude_rates(0.0, 0.0, 0.0, 0.0)
            now = time.time()
            if now - self.last_arm_wall >= 0.35:
                self.mavlink.arm(True)
                self.last_arm_wall = now
            self.localizer.update_async(0)
            time.sleep(min(self.dt, 0.01))
        raise RuntimeError(
            "simulator race clock did not release the official countdown"
        )

    def _observe(self, state) -> np.ndarray:
        return build_observation(
            position_world=state.position,
            quat_wxyz=state.quat_wxyz,
            velocity_world=state.velocity,
            gyro_raw=state.gyro_raw,
            previous_action=self.previous_action,
            gate_index=min(self.target, N_RACE_GATES - 1),
            geometry=self.geometry,
            position_sigma_m=state.position_sigma_m,
        )

    def reset(self) -> tuple[np.ndarray, dict]:
        self.localizer.stop_async()
        self.emergency_stop()
        self._hard_reset_to_countdown()
        self.last_arm_wall = 0.0
        # Anchor briefly while physically disarmed, then hold zero thrust until
        # the simulator's official race clock releases the vehicle.
        try:
            state = self.localizer.initialize(
                timeout_s=self.config.anchor_timeout_s,
                keepalive=None,
            )
        except RuntimeError as error:
            if "spawn gate anchor failed:" not in str(error):
                raise
            # A single GateNet inference can occasionally consume most of the
            # short launch window while the simulator and learner share the
            # GPU.  Keep the drone disarmed and collect one brief fallback
            # batch instead of killing the entire training process.
            state = self.localizer.initialize(
                timeout_s=max(1.25, self.config.anchor_timeout_s),
                keepalive=None,
            )
        if self.localizer.anchor_diagnostics.get(
            "translation_spread_p90_m", np.inf
        ) > 0.25:
            raise RuntimeError(
                "spawn visual anchor is inconsistent: "
                f"{self.localizer.anchor_diagnostics}"
            )
        spawn_pitch = float(
            self.localizer.anchor_diagnostics.get("pitch_deg", np.inf)
        )
        if not -30.0 <= spawn_pitch <= -8.0:
            raise RuntimeError(
                "spawn attitude does not match the known pitched pad: "
                f"{self.localizer.anchor_diagnostics}"
            )

        for _ in range(8):
            self.mavlink.arm(True)
            self.mavlink.send_attitude_rates(0.0, 0.0, 0.0, 0.0)
            time.sleep(0.01)
        self.last_arm_wall = time.time()
        status = self.mavlink.race_status
        if not (_race_pending(status) or _race_active(status)):
            raise RuntimeError(
                "VQ2 race is neither pending nor active after spawn anchor"
            )
        self.target = int(status["active_gate"])
        if self.target != 0:
            raise RuntimeError(
                f"new VQ2 episode began at stale gate {self.target}"
            )
        self.localizer.start_async()
        state = self.localizer.update_async(self.target)
        countdown_hold_s = 0.0
        if self.config.wait_for_official_start:
            status, countdown_hold_s = self._wait_for_official_release()
            state = self.localizer.update_async(self.target)
        self.spawn_position = state.position.copy()
        self._configure_anchored_course()
        now = time.time()
        self.previous_action[:] = 0.0
        self.episode_started_wall = now
        self.last_step_wall = now
        self.gate_started_wall = now
        if not self.mavlink.imu:
            raise RuntimeError("no IMU clock at VQ2 race start")
        self.last_sim_us = int(self.mavlink.imu[-1][0])
        self.next_sim_us = self.last_sim_us
        self.episode_started_sim = self.last_sim_us * 1e-6
        self.gate_started_sim = self.episode_started_sim
        self.best_progress_sim = self.episode_started_sim
        self.collision_after_wall_ns = time.time_ns()
        self.previous_progress = course_progress(
            state.position, self.target, self.geometry, self.spawn_position
        )
        self.episode_return = 0.0
        self.best_progress = self.previous_progress
        self.best_progress_wall = now
        self.previous_signed_plane = float(np.dot(
            state.position - self.geometry.positions[0],
            self.gate_normals[0],
        ))
        self.pending_miss_sim = None
        self.overspeed_started_sim = None
        self.localization_bad_since = None
        self.predictive_release_pending = bool(
            status.get("predictive_release", False)
        )
        self.last_status_boot_ms = int(
            status.get("source_sim_boot_ms", status["sim_boot_ms"])
        )
        self.race_was_active = bool(
            _race_active(status) and not self.predictive_release_pending
        )
        self.steps = 0
        info = {
            "target": self.target,
            "anchor": dict(self.localizer.anchor_diagnostics),
            "countdown_hold_s": countdown_hold_s,
            "race_start_ms": int(status["race_start_ms"]),
            "release_sim_boot_ms": int(status["sim_boot_ms"]),
            "predictive_release": bool(
                status.get("predictive_release", False)
            ),
        }
        return self._observe(state), info

    def _new_collisions(self) -> list[tuple]:
        rows = self.mavlink.collisions_since(self.collision_after_wall_ns)
        # VQ2 emits harmless low-impulse id=1002/threat=1 pad-contact
        # packets while the freshly armed drone is still separating from the
        # tilted start pad.  They are not crashes and previously caused
        # spurious 7-12-step terminal episodes.  Preserve every material
        # contact as an immediate terminal event.
        return [
            row for row in rows
            if not (
                row[1] == 1002
                and row[2] == 1
                and row[3] < 0.20
            )
        ]

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        step_started = time.perf_counter()
        config = self.config
        action = np.clip(
            np.asarray(action, np.float32), -1.0, 1.0
        ).copy()
        command = action_to_wire_command(action)
        command[3] = min(float(command[3]), config.thrust_cap)
        launch_elapsed = max(
            self.last_sim_us * 1e-6 - self.episode_started_sim,
            0.0,
        )
        launch_assist = bool(
            self.target == 0
            and launch_elapsed < config.launch_assist_s
        )
        if launch_assist:
            command[3] = max(
                float(command[3]), config.launch_min_thrust
            )
            action[3] = 2.0 * float(command[3]) - 1.0
        self.mavlink.send_attitude_rates(*map(float, command))
        now = time.time()
        if now - self.last_arm_wall >= 0.35:
            self.mavlink.arm(True)
            self.last_arm_wall = now

        # The simulator's GPU rendering can pause wall-clock physics near a
        # gate.  Pace policy transitions by HIGHRES_IMU sim time so actions
        # cannot race ahead while the simulated drone is frozen.
        step_us = int(round(self.dt * 1_000_000.0))
        self.next_sim_us += step_us
        sim_wait_deadline = time.time() + 1.5
        sim_clock_stalled = False
        while (
            time.time() < sim_wait_deadline
            and (
                not self.mavlink.imu
                or int(self.mavlink.imu[-1][0]) < self.next_sim_us
            )
        ):
            now_wait = time.time()
            if now_wait - self.last_arm_wall >= 0.35:
                self.mavlink.arm(True)
                self.last_arm_wall = now_wait
            time.sleep(0.001)
        if (
            not self.mavlink.imu
            or int(self.mavlink.imu[-1][0]) < self.next_sim_us
        ):
            sim_clock_stalled = True

        old_target = self.target
        status = self.mavlink.race_status
        reported_target = int(status["active_gate"]) if status else old_target
        if (
            self.predictive_release_pending
            and status
            and _race_active(status)
        ):
            self.predictive_release_pending = False
            self.race_was_active = True
        status_boot_regressed = bool(
            status
            and not self.predictive_release_pending
            and self.last_status_boot_ms is not None
            and int(status["sim_boot_ms"]) + 100
            < self.last_status_boot_ms
        )
        race_reentered_countdown = bool(
            status
            and not self.predictive_release_pending
            and self.race_was_active
            and _race_pending(status)
        )
        if status:
            self.last_status_boot_ms = int(status["sim_boot_ms"])
            self.race_was_active = (
                self.race_was_active or _race_active(status)
            )
        try:
            localizer_started = time.perf_counter()
            state = self.localizer.update_async(
                min(reported_target, N_RACE_GATES - 1)
            )
            localizer_ms = (
                time.perf_counter() - localizer_started
            ) * 1000.0
            localizer_error = None
        except Exception as error:  # terminal sensor failure, logged verbatim
            state = self.localizer.state()
            localizer_ms = (
                time.perf_counter() - localizer_started
            ) * 1000.0
            localizer_error = repr(error)
        now = time.time()
        current_sim_us = (
            int(self.mavlink.imu[-1][0])
            if self.mavlink.imu else self.last_sim_us
        )
        now_sim = current_sim_us * 1e-6
        elapsed_sim = float(max(
            (current_sim_us - self.last_sim_us) * 1e-6,
            0.0,
        ))
        self.steps += 1

        target_advance = reported_target - old_target
        self.target = max(self.target, reported_target)
        gate_event_correction = None
        official_gate_time_s = None
        gate_event_age_s = None
        if status and int(status.get("last_gate_time", -1)) >= 0:
            official_gate_time_s = (
                int(status["last_gate_time"]) * 1e-9
            )
        if (
            self.gate_event_plane_correction_enabled
            and target_advance == 1
            and official_gate_time_s is not None
        ):
            race_elapsed_s = max(
                (
                    int(status["sim_boot_ms"])
                    - int(status["race_start_ms"])
                ) * 1e-3,
                0.0,
            )
            gate_event_age_s = float(np.clip(
                race_elapsed_s - official_gate_time_s,
                0.0,
                0.5,
            ))
            plane_speed_mps = max(float(np.dot(
                state.velocity, self.gate_normals[old_target]
            )), 0.0)
            gate_event_correction = self.localizer.apply_gate_plane_event(
                old_target,
                forward_offset_m=(
                    plane_speed_mps * gate_event_age_s
                    + config.gate_event_forward_offset_m
                ),
            )
            if gate_event_correction is not None:
                gate_event_correction.update({
                    "event_age_s": gate_event_age_s,
                    "plane_speed_mps": plane_speed_mps,
                })
            state = self.localizer.state()
        finished = bool(
            status
            and (
                status["race_finish_ns"] >= 0
                or reported_target >= N_RACE_GATES
            )
        )
        progress_target = min(self.target, N_RACE_GATES - 1)
        progress = course_progress(
            state.position,
            progress_target,
            self.geometry,
            self.spawn_position,
        )
        delta_progress = float(np.clip(
            progress - self.previous_progress, -0.5, 1.0
        ))
        reward = (
            config.progress_scale * delta_progress
            + config.gate_bonus * max(target_advance, 0)
            - config.time_penalty_per_s * elapsed_sim
            - config.action_smoothness
            * float(np.sum((action - self.previous_action) ** 2))
        )

        if target_advance > 0:
            self.gate_started_wall = now
            self.gate_started_sim = now_sim
            self.best_progress = progress
            self.best_progress_wall = now
            self.best_progress_sim = now_sim
            self.previous_signed_plane = None
            self.pending_miss_sim = None
        elif progress > self.best_progress + 0.05:
            self.best_progress = progress
            self.best_progress_wall = now
            self.best_progress_sim = now_sim

        terminated = False
        truncated = False
        failure = None
        plane_cross_without_event = False
        collisions = self._new_collisions()
        if not finished:
            if collisions:
                failure = "collision"
                reward -= config.collision_penalty
            elif localizer_error is not None:
                failure = "localizer_exception"
                reward -= config.failed_run_penalty
            elif status_boot_regressed or race_reentered_countdown:
                failure = "simulator_respawn"
                reward -= config.failed_run_penalty
            elif target_advance < 0 or target_advance > 1:
                failure = "invalid_gate_sequence"
                reward -= config.failed_run_penalty
            elif sim_clock_stalled:
                failure = "sim_clock_stalled"
                reward -= config.failed_run_penalty

        # Crossing a target's physical plane without an official gate event is
        # a miss.  Give MAVLink 220 ms to deliver the authoritative event.
        if not finished and self.target < N_RACE_GATES:
            signed = float(np.dot(
                state.position - self.geometry.positions[self.target],
                self.gate_normals[self.target],
            ))
            if (
                target_advance == 0
                and self.previous_signed_plane is not None
                and self.previous_signed_plane < 0.0 <= signed
            ):
                self.pending_miss_sim = now_sim
            self.previous_signed_plane = signed
            if (
                self.pending_miss_sim is not None
                and now_sim - self.pending_miss_sim
                >= config.miss_confirmation_s
                and target_advance == 0
            ):
                # EKF geometry is not authoritative in VQ2. A position error
                # can make the estimated trajectory cross the gate plane even
                # while the real drone remains on the approach. Keep targeting
                # the same gate until the simulator reports a pass, a genuine
                # collision occurs, or the normal gate timeout expires.
                plane_cross_without_event = True
                self.pending_miss_sim = None

        visual_bad = (
            state.visual_age_s > config.max_visual_age_s
            or state.position_sigma_m > config.max_position_sigma_m
            or not np.all(np.isfinite(state.position))
        )
        if visual_bad:
            self.localization_bad_since = self.localization_bad_since or now
        else:
            self.localization_bad_since = None
        if (
            not finished
            and
            self.localization_bad_since is not None
            and now - self.localization_bad_since
            >= config.localization_grace_s
        ):
            failure = failure or "localization_lost"
            reward -= config.failed_run_penalty

        speed = float(np.linalg.norm(state.velocity))
        if speed > config.speed_cap_mps:
            if self.overspeed_started_sim is None:
                self.overspeed_started_sim = now_sim
        elif speed < (
            config.speed_cap_mps - config.overspeed_hysteresis_mps
        ):
            self.overspeed_started_sim = None
        sustained_overspeed = bool(
            self.overspeed_started_sim is not None
            and now_sim - self.overspeed_started_sim
            >= config.overspeed_confirmation_s
        )
        off_track = _distance_to_polyline(
            state.position, self.track_points
        )
        qw, qx, qy, qz = state.quat_wxyz
        body_down = Rotation.from_quat(
            [qx, qy, qz, qw]
        ).as_matrix()[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(
            body_down[2], -1.0, 1.0
        ))))
        distance = (
            float(np.linalg.norm(
                self.geometry.positions[self.target] - state.position
            ))
            if self.target < N_RACE_GATES else 0.0
        )
        no_progress_limit = (
            config.near_no_progress_s if distance < 8.0
            else config.far_no_progress_s
        )
        if not finished and sustained_overspeed:
            failure = failure or "overspeed"
            reward -= config.failed_run_penalty
        elif not finished and off_track > config.max_off_track_m:
            failure = failure or "off_track"
            reward -= config.failed_run_penalty
        elif not finished and tilt > config.max_tilt_deg:
            failure = failure or "inverted"
            reward -= config.failed_run_penalty
        elif (
            not finished
            and now_sim - self.gate_started_sim > config.gate_timeout_s
        ):
            failure = failure or "gate_timeout"
            reward -= config.failed_run_penalty
        elif (
            not finished
            and now_sim - self.best_progress_sim > no_progress_limit
        ):
            failure = failure or "no_progress"
            reward -= config.failed_run_penalty

        camera_age = (
            (time.time_ns() - self.localizer.vision.latest[3]) * 1e-9
            if self.localizer.vision.latest is not None else np.inf
        )
        imu_age = (
            (time.time_ns() - self.mavlink.imu[-1][-1]) * 1e-9
            if self.mavlink.imu else np.inf
        )
        timing_healthy = bool(
            finished
            or (
                not sim_clock_stalled
                and elapsed_sim <= config.max_transition_sim_s
                and camera_age <= config.max_camera_packet_age_s
                and imu_age <= config.max_imu_packet_age_s
            )
        )
        if not timing_healthy:
            if failure is None:
                failure = "stale_sensor_stream"
                reward -= config.failed_run_penalty

        if finished:
            reward += config.finish_bonus
            terminated = True
        elif failure is not None:
            # Keep terminal rewards Markov: the penalty is determined by the
            # current failure event, while dense progress earned earlier in
            # the episode remains intact.  Clamping the whole episode return
            # here made two different approaches to the same gate
            # indistinguishable to the critic and could turn useful progress
            # into an extra terminal penalty.
            terminated = True
        elif (
            now_sim - self.episode_started_sim
            >= config.max_episode_s
        ):
            truncated = True

        self.previous_progress = progress
        self.episode_return += reward
        self.previous_action = action.copy()
        self.last_step_wall = now
        self.last_sim_us = current_sim_us
        observation = self._observe(state)
        info = {
            "target": self.target,
            "gates_passed": max(target_advance, 0),
            "finished": finished,
            "failure": failure,
            "plane_cross_without_event": plane_cross_without_event,
            "gate_event_plane_correction": gate_event_correction,
            "gate_event_age_s": gate_event_age_s,
            "official_gate_time_s": official_gate_time_s,
            "speed": speed,
            "overspeed_duration_s": (
                max(0.0, now_sim - self.overspeed_started_sim)
                if self.overspeed_started_sim is not None else 0.0
            ),
            "off_track": off_track,
            "tilt_deg": tilt,
            "position_sigma_m": state.position_sigma_m,
            "visual_age_s": state.visual_age_s,
            "camera_age_s": camera_age,
            "imu_age_s": imu_age,
            "sim_time_s": now_sim,
            "sim_step_s": elapsed_sim,
            "timing_healthy": timing_healthy,
            "corners_fused": state.corners_fused,
            "localizer_source": self.localizer.last_update_source,
            "vision_inference_ms": self.localizer.async_inference_ms,
            "crop_track_ms": self.localizer.crop_track_ms,
            "crop_tracker_enabled": self.localizer.crop_track_enabled,
            "crop_track_updates": self.localizer.update_counts.get(
                "crop_track", 0
            ),
            "position": state.position.tolist(),
            "action": action.tolist(),
            "wire_command": command.tolist(),
            "launch_assist": launch_assist,
            "localizer_error": localizer_error,
            "localizer_source": self.localizer.last_update_source,
            "vision_inference_ms": self.localizer.async_inference_ms,
            "localizer_ms": localizer_ms,
            "step_ms": (
                time.perf_counter() - step_started
            ) * 1000.0,
        }
        if terminated:
            self._request_immediate_respawn()
        return observation, float(reward), terminated, truncated, info
