"""Massively parallel VQ2 track environment (torch).

Dynamics: the identified surrogate (aigp/fastsim/sysid fit), batched.
Track: the click-certified map (vq2_map_final.json) with real gate frame
geometry; collision is evaluated analytically at gate-plane crossings.
Observations: the live stack's 53-D layout (aigp.rl.vq2_features), with
an injected estimator-error model so the policy trains against the kind
of state it will actually receive from the V7/V11 localizer.

Frame note: simulated in the course/EKF frame (z down, g=+9.81 z). The
rate-loop gain sign is exposed as a config so the open-loop attitude
check in scripts/fastsim_train_ppo.py can pin it against the real lap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from aigp.fastsim.sysid import SurrogateModel

N_GATES = 17
N_LOOKAHEAD = 3
OBS_DIM = 53
ACT_DIM = 4
MAX_RATE = float(np.deg2rad(180.0))
MAX_YAW_RATE = float(np.deg2rad(120.0))
RATE_GAIN_LIVE = np.array([2.33, 2.35, 2.36], dtype=np.float32)
WIRE_RATE_LIMIT = np.array(
    [MAX_RATE / RATE_GAIN_LIVE[0], MAX_RATE / RATE_GAIN_LIVE[1],
     MAX_YAW_RATE / RATE_GAIN_LIVE[2]],
    dtype=np.float32,
)
HOLE_HALF = 0.75
PANEL_HALF = 1.36


@dataclass
class FastEnvConfig:
    n_envs: int = 4096
    race_gates: int = N_GATES
    control_hz: float = 30.0
    substeps: int = 4
    max_episode_s: float = 45.0
    # reward
    progress_scale: float = 2.0
    gate_bonus: float = 25.0
    finish_bonus: float = 600.0
    # Optional full-start-only racing-time objective.  Random curriculum
    # starts reset the episode clock and therefore cannot be compared on
    # absolute finish time; ``spawn_flag`` gates this term to real starts.
    finish_time_target_s: float = 0.0
    finish_time_bonus_per_s: float = 0.0
    # Optional segment racing objective, paid only on an authoritative valid
    # gate crossing.  Targets are elapsed seconds since the previous gate.
    gate_time_targets_s: tuple = ()
    gate_time_bonus_per_s: float = 0.0
    # crashing must NOT dominate timing out, or the policy learns that
    # gates are lava and steers around every plane crossing
    collision_penalty: float = 30.0
    time_penalty_per_s: float = 0.8
    action_smoothness: float = 0.02
    clearance_bonus: float = 6.0       # scaled by hole-center clearance
    corridor_m: float = 8.0
    # tube around the DEMONSTRATED path (which threads all the real
    # scenery -- pillars/jets the surrogate cannot model). When demo
    # states exist, the corridor uses the flown line at this radius.
    demo_corridor_m: float = 2.0
    # Dense, opt-in preference for the demonstrated racing line.  The free
    # radius preserves room for local speed optimization while making large
    # geometric shortcuts expensive.  Units are reward / (m^2 * second).
    demo_tracking_penalty_per_s: float = 0.0
    demo_tracking_free_m: float = 0.15
    # At a valid gate crossing, reward proximity to the demo's crossing
    # point.  This preserves setup offsets (notably gate 1 -> gate 2) that a
    # center-clearance reward alone incorrectly erases.
    demo_crossing_bonus: float = 0.0
    demo_crossing_radius_m: float = 0.50
    offtrack_penalty: float = 30.0
    gate_timeout_s: float = 6.0
    # estimator noise (OU) ranges, sampled per episode -- matched to the
    # MEASURED EKF (3-10 cm through-gate, tens of cm between gates)
    pos_noise_lo: float = 0.02
    pos_noise_hi: float = 0.15
    pos_noise_tau_s: float = 1.2
    att_noise_deg_hi: float = 2.0
    # structured estimator error (measured on the real certified lap):
    # every ~6-14s the belief drifts for 0.3-1.2s up to 0.4-1.5m, then
    # SNAPS back in one step (the relocalization correction)
    reloc_events: bool = False
    reloc_interval_s: tuple = (5.0, 12.0)
    # vision droughts run seconds in the real stack (landmark age up to
    # ~3 s between gates); the policy must fly through them on momentum
    reloc_drift_s: tuple = (0.4, 3.0)
    reloc_mag_m: tuple = (0.4, 1.6)
    # continuous speed-proportional coast drift coefficients (diffuse,
    # directional-bias); 3 Hz-era defaults preserved for reproducibility
    coast_speed_diffuse: float = 0.004
    coast_speed_bias: float = 0.010
    # FOV-coupled vision (flight-6/7 root cause): fixes only land when a
    # lookahead gate sits inside the tilted camera frustum, so a policy
    # that sprints pitched-forward blinds itself and feels the drift.
    # Off by default for legacy-config reproducibility.
    fov_vision: bool = False
    fov_cam_pitch_deg: float = 20.0
    fov_half_h_deg: float = 43.0       # 90/58.7 deg frustum minus margin
    fov_half_v_deg: float = 27.0
    fov_max_range_m: float = 45.0
    fov_fix_rate_hz: float = 10.0
    fov_detect_prob: float = 0.85
    fov_coast_after_s: float = 0.25
    # Optional learned state-dependent fusion probability.  Empty preserves
    # the legacy constant Bernoulli model.
    vision_outcome_model: str = ""
    # When set by batched candidate search, stochastic detector draws are
    # shared across candidates for each world instead of rewarding lucky
    # candidate-specific random streams.
    common_random_worlds: int = 0
    # Number of upcoming mapped gates allowed to supply a landmark fix.
    # The legacy/current-gate path sees active + one lookahead. Course-wide
    # multigate association can use any plausibly visible mapped gate.
    fov_gate_lookahead: int = 2
    # live-guard parity (VQ2LiveEnv terminals mirrored)
    max_tilt_deg: float = 80.0
    live_gate_timeout_s: float = 4.5
    no_progress_s: float = 1.5
    # domain randomization ranges (multipliers)
    dr_thrust: tuple = (0.85, 1.15)
    dr_rate_gain: tuple = (0.70, 1.15)
    dr_rate_tau: tuple = (0.7, 1.4)
    dr_drag: tuple = (0.1, 0.6)
    act_delay_steps_max: int = 2
    # measured plant lag is ~30ms (17ms dead + 13ms tau, lrspeiser
    # open-loop); a zero-delay draw lets policies learn twitch control
    # that arrives late in reality (the violent-flight symptom)
    act_delay_steps_min: int = 0
    rate_gain_sign: float = 1.0        # pinned by attitude check
    thrust_wire_cap: float = 0.52      # live stack caps wire thrust
    random_start_frac: float = 0.6
    # Optional target-gate weights for demonstrated-state curriculum starts.
    # Empty preserves the legacy gate-0/1 launch weighting.
    demo_gate_weights: tuple = ()
    start_noise_pos_m: float = 0.4
    # live episodes begin AT REST on the pitched pad; the legacy
    # "just-lifted" spawn (v ~ 2 m/s climb) left the at-rest state
    # out-of-distribution -- the sharper corrected-math policies
    # deterministically crashed in the first 1.8s live (obs-divergence
    # diagnosis, t=0 vbody mismatch)
    spawn_at_rest: bool = False
    # residual authority when a backbone is attached
    residual_scale: float = 0.25
    # Empty means all gates.  Used to audit/deploy a learned correction only
    # where it transfers better than the protected reference controller.
    residual_active_gates: tuple = ()
    residual_gate_scales: tuple = ()
    start_noise_vel_mps: float = 0.8
    speed_cap_mps: float = 16.0
    # Optional learned correction to the analytic plant. Each simulated world
    # is assigned one ensemble member, so optimization must work across model
    # disagreement instead of exploiting the ensemble mean.
    world_model_residual_scale: float = 1.0
    world_model_aleatoric_scale: float = 0.0
    world_model_use_mean: bool = False
    # Sparse physical kicks broaden the recovery-state distribution without
    # replacing the vehicle model with unstructured per-step noise.  A rate
    # of 0.06 Hz gives roughly one disturbed five-gate lap in two.
    impulse_rate_hz: float = 0.0
    impulse_velocity_mps: tuple = (0.10, 0.55)
    impulse_vertical_scale: float = 0.35
    impulse_start_s: float = 0.8

    def apply_vision10hz(self) -> "FastEnvConfig":
        """Retune estimator noise to the GPU-10Hz vision era.

        Measured on v77 (vision_hz 10, cuda; 49 episodes, 12.4k steps):
        sigma p50/p90/p99 = 0.09/0.16/0.43 m; landmark age p50/p90/p99 =
        0.14/0.36/1.07 s; belief snaps p50/p90/max = 0.12/0.25/1.19 m;
        coast spells p50/p90/p99 = 0.19/0.24/0.89 s.  The 3 Hz-era
        defaults model 4x longer droughts and 4x larger snaps than the
        current stack produces.
        """
        self.pos_noise_lo = 0.03
        self.pos_noise_hi = 0.16
        self.reloc_interval_s = (6.0, 14.0)
        self.reloc_drift_s = (0.15, 0.9)
        self.reloc_mag_m = (0.10, 0.5)
        self.coast_speed_diffuse = 0.002
        self.coast_speed_bias = 0.004
        return self

    def apply_multigate10hz(self) -> "FastEnvConfig":
        """Vision-10Hz noise with course-wide landmark availability.

        Keep detector probability, OU noise, coast drift, and relocation
        magnitudes unchanged.  The only modeled upgrade is the validated
        association behavior: any mapped gate in the camera frustum may
        refresh the filter instead of only the active gate and one lookahead.
        """
        self.apply_vision10hz()
        self.fov_gate_lookahead = N_GATES
        return self


class FastVQ2Env:
    def __init__(
        self,
        model: SurrogateModel,
        map_path: str | Path,
        demo_states: dict | None = None,
        config: FastEnvConfig | None = None,
        device: str = "cuda",
        obstacles_path: str | Path | None = None,
        backbone=None,
        residual_ensemble=None,
        spawn_states: dict | None = None,
    ) -> None:
        # residual mode: when a RefController backbone is attached, the
        # incoming action is a RESIDUAL added to the backbone's action
        self.backbone = backbone
        self.residual_ensemble = residual_ensemble
        self.spawn_states = None
        self.cfg = config or FastEnvConfig()
        self.device = torch.device(device)
        self.model = model
        cfg = self.cfg
        n = cfg.n_envs

        gates = json.loads(Path(map_path).read_text())["gates"]
        pos = np.array([g["pos"] for g in gates[:N_GATES]], np.float32)
        quats = np.array(
            [g["quat_wxyz"] for g in gates[:N_GATES]], np.float32
        )
        from scipy.spatial.transform import Rotation
        R = Rotation.from_quat(
            np.stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]],
                     axis=1)
        ).as_matrix().astype(np.float32)
        normals = R[:, :, 1].copy()
        # orient each gate frame so local +y points along course direction
        # (180-deg z-rotation of the gate frame; the hole square is
        # symmetric under it, and the crossing test assumes -y -> +y)
        for i in range(N_GATES):
            incoming = pos[i] - (pos[i - 1] if i else np.zeros(3,
                                                              np.float32))
            if np.dot(normals[i], incoming) < 0:
                normals[i] = -normals[i]
                R[i, :, 0] = -R[i, :, 0]
                R[i, :, 1] = -R[i, :, 1]
        t = torch.tensor
        self.gate_pos = t(pos, device=self.device)
        self.gate_R = t(R, device=self.device)          # (G,3,3)
        self.gate_normal = t(normals, device=self.device)
        gate_time_targets = np.zeros(N_GATES, np.float32)
        if cfg.gate_time_targets_s:
            values = np.asarray(cfg.gate_time_targets_s, np.float32)
            gate_time_targets[:min(len(values), N_GATES)] = values[:N_GATES]
        self.gate_time_targets = t(gate_time_targets, device=self.device)
        tangents = np.zeros_like(pos)
        for i in range(N_GATES):
            a = pos[max(0, i - 1)]
            b = pos[min(N_GATES - 1, i + 1)]
            d = b - a
            tangents[i] = d / (np.linalg.norm(d) + 1e-9)
        self.gate_tangent = t(tangents.astype(np.float32),
                              device=self.device)
        spawn = np.array([0.0, 0.0, -0.3], np.float32)
        self.track_points = t(
            np.vstack([spawn, pos]).astype(np.float32), device=self.device
        )
        seg = np.diff(np.vstack([spawn, pos]), axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        self.seg_len = t(seg_len.astype(np.float32), device=self.device)
        self.cum_len = t(
            np.concatenate([[0.0], np.cumsum(seg_len)]).astype(np.float32),
            device=self.device,
        )

        # demo states for random starts: dict with pos/vel/quat arrays plus
        # per-row current-gate index
        self.demo = None
        if demo_states is not None:
            self.demo = {
                k: t(np.asarray(v, np.float32), device=self.device)
                for k, v in demo_states.items()
            }
            # launch curriculum: overweight early-course demo states so
            # the parked->flying transition is rehearsed constantly
            gate_arr = self.demo["gate"]
            if cfg.demo_gate_weights:
                table = torch.as_tensor(
                    cfg.demo_gate_weights, dtype=torch.float32,
                    device=self.device,
                )
                row = torch.clamp(
                    gate_arr.long(), 0, len(cfg.demo_gate_weights) - 1
                )
                w = table[row]
            else:
                w = 1.0 + 5.0 * (gate_arr <= 1).float()
            self.demo_weights = w / w.sum()
            # "spawn" start = just-lifted state (the live stack's scripted
            # launch assist owns pad separation; the policy's job begins
            # airborne). Pick the first demo row past 1.5 m/s on gate 0.
            speed = torch.linalg.norm(self.demo["vel"], dim=1)
            cand = torch.nonzero(
                (self.demo["gate"] == 0) & (speed > 1.5)
            ).squeeze(-1)
            k0 = int(cand[0]) if len(cand) else 0
            self.launch_state = {
                "pos": self.demo["pos"][k0].clone(),
                "vel": self.demo["vel"][k0].clone(),
                "quat": self.demo["quat"][k0].clone(),
            }
        self.spawn_flag = torch.zeros(
            cfg.n_envs, dtype=torch.bool, device=self.device
        )
        # authoritative obstacle cylinders (cooked actor roots): kill on
        # XY proximity while within the obstacle's vertical span
        self.obstacles = None
        if obstacles_path is not None and Path(obstacles_path).exists():
            ob = json.loads(Path(obstacles_path).read_text())
            rows = []
            for group in ("airplanes", "stations"):
                for o in ob.get(group, []):
                    rows.append([*o["pos"], o["radius"], o["height"]])
            if rows:
                arr = np.asarray(rows, np.float32)
                self.obs_xy = t(arr[:, :2], device=self.device)
                self.obs_z_root = t(arr[:, 2], device=self.device)
                self.obs_r = t(arr[:, 3], device=self.device)
                self.obs_h = t(arr[:, 4], device=self.device)
                self.obstacles = True
        # demo-path corridor segments: precomputed (multi-path npz with
        # explicit segment arrays) or derived from the pos stream
        self.demo_path = None
        if self.demo is not None and "path_a" in self.demo:
            self.demo_path_a = self.demo["path_a"]
            self.demo_path_d = self.demo["path_d"]
            self.demo_path_len2 = (
                self.demo_path_d * self.demo_path_d
            ).sum(-1).clamp(min=1e-9)
            self.demo_path = True
        elif self.demo is not None and len(self.demo["pos"]) > 10:
            dp = self.demo["pos"][::3]           # ~10 Hz spacing
            self.demo_path_a = dp[:-1]
            self.demo_path_d = dp[1:] - dp[:-1]
            self.demo_path_len2 = (
                self.demo_path_d * self.demo_path_d
            ).sum(-1).clamp(min=1e-9)
            self.demo_path = True

        # Recover each demonstrated gate-plane crossing in local gate x/z.
        # Prefer an actual consecutive sign change; fall back to the closest
        # sample on that gate's approach when packet/event timing skips it.
        self.demo_cross_local = torch.zeros(
            N_GATES, 2, dtype=torch.float32, device=self.device
        )
        self.demo_cross_valid = torch.zeros(
            N_GATES, dtype=torch.bool, device=self.device
        )
        if self.demo is not None:
            demo_p = self.demo["pos"]
            demo_gate = self.demo["gate"].long()
            for gate_idx in range(min(cfg.race_gates, N_GATES)):
                rel = demo_p - self.gate_pos[gate_idx]
                local = torch.einsum(
                    "ij,ni->nj", self.gate_R[gate_idx], rel
                )
                approach = demo_gate == gate_idx
                crossing = torch.nonzero(
                    approach[:-1]
                    & (local[:-1, 1] <= 0.0)
                    & (local[1:, 1] >= 0.0)
                ).squeeze(-1)
                if len(crossing):
                    k = int(crossing[-1])
                    y0, y1 = local[k, 1], local[k + 1, 1]
                    frac = torch.clamp(
                        -y0 / (y1 - y0 + 1e-9), 0.0, 1.0
                    )
                    point = local[k] + frac * (local[k + 1] - local[k])
                else:
                    rows = torch.nonzero(approach).squeeze(-1)
                    if not len(rows):
                        continue
                    k = rows[torch.argmin(local[rows, 1].abs())]
                    point = local[k]
                self.demo_cross_local[gate_idx] = point[[0, 2]]
                self.demo_cross_valid[gate_idx] = True

        z = lambda *shape: torch.zeros(*shape, device=self.device)
        if spawn_states is not None:
            self.spawn_states = {
                key: torch.as_tensor(value, dtype=torch.float32,
                                     device=self.device)
                for key, value in spawn_states.items()
            }
        self.p = z(n, 3)
        self.v = z(n, 3)
        self.q = z(n, 4)                     # wxyz
        self.w = z(n, 3)
        self.target = torch.zeros(n, dtype=torch.long, device=self.device)
        self.t_ep = z(n)
        self.t_gate = z(n)
        self.prev_action = z(n, ACT_DIM)
        self.act_buf = z(cfg.act_delay_steps_max + 1, n, ACT_DIM)
        self.act_delay = torch.zeros(n, dtype=torch.long,
                                     device=self.device)
        self.noise_pos = z(n, 3)
        self.noise_amp = z(n, 1)
        self.vis_age = z(n)
        self.reloc_next_t = z(n)
        self.reloc_end_t = z(n)
        self.reloc_dir = z(n, 3)
        self.reloc_rate = z(n)
        self.best_prog = z(n)
        self.t_best = z(n)
        self.progress = z(n)
        # DR params
        self.dr_thrust = z(n, 1)
        self.dr_K = z(n, 3)
        self.dr_tau = z(n, 3)
        self.dr_drag = z(n, 3)
        self.residual_member = torch.zeros(
            n, dtype=torch.long, device=self.device
        )
        self.world_model_disagreement = z(n)
        self.world_model_support_z = z(n)
        self.vision_prev_fix = z(n)
        self.vision_outcome = None
        if cfg.vision_outcome_model:
            outcome = json.loads(Path(cfg.vision_outcome_model).read_text())
            if outcome.get("type") != "vq2_vision_fusion_logistic_v1":
                raise ValueError("unsupported vision outcome model")
            if len(outcome["weight"]) != 21:
                raise ValueError("vision outcome model must have 21 features")
            self.vision_outcome = {
                "mean": t(outcome["feature_mean"], device=self.device),
                "std": t(outcome["feature_std"], device=self.device),
                "weight": t(outcome["weight"], device=self.device),
                "bias": float(outcome["bias"]),
            }
        self.reset(torch.arange(n, device=self.device))

    # ---------- quaternion helpers (wxyz) ----------
    @staticmethod
    def _qmul(a, b):
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack([
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ], dim=-1)

    @staticmethod
    def _qrotvec(w, dt):
        angle = torch.linalg.norm(w, dim=-1, keepdim=True) * dt
        axis = w / (torch.linalg.norm(w, dim=-1, keepdim=True) + 1e-9)
        half = angle * 0.5
        return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)

    @staticmethod
    def _qmat(q):
        w, x, y, z = q.unbind(-1)
        return torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
                         2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
                         2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w),
                         1 - 2 * (x * x + y * y)], -1),
        ], dim=-2)

    def reset(self, idx: torch.Tensor) -> None:
        cfg = self.cfg
        n = len(idx)
        if n == 0:
            return
        dev = self.device
        use_demo = (
            self.demo is not None
            and torch.rand(n, device=dev) < cfg.random_start_frac
        )
        # default "spawn" start: just-lifted post-launch-assist state
        if cfg.spawn_at_rest:
            p = torch.tensor([0.0, 0.0, -0.05], device=dev).repeat(n, 1) \
                + torch.randn(n, 3, device=dev) * 0.02
            v = torch.zeros(n, 3, device=dev)
            pitch = torch.tensor(-17.8 * np.pi / 360.0, device=dev)
            q = torch.zeros(n, 4, device=dev)
            q[:, 0] = torch.cos(pitch)
            q[:, 2] = torch.sin(pitch)
        elif self.demo is not None:
            p = self.launch_state["pos"].repeat(n, 1) \
                + torch.randn(n, 3, device=dev) * 0.3
            v = self.launch_state["vel"].repeat(n, 1) \
                + torch.randn(n, 3, device=dev) * 0.4
            q = self.launch_state["quat"].repeat(n, 1)
        else:
            p = torch.tensor([0.0, 0.0, -0.8], device=dev).repeat(n, 1)
            v = torch.zeros(n, 3, device=dev)
            pitch = torch.tensor(-17.8 * np.pi / 360.0, device=dev)
            q = torch.zeros(n, 4, device=dev)
            q[:, 0] = torch.cos(pitch)
            q[:, 2] = torch.sin(pitch)
        if self.spawn_states is not None:
            sample = torch.randint(
                len(self.spawn_states["pos"]), (n,), device=dev
            )
            p = self.spawn_states["pos"][sample].clone()
            v = self.spawn_states["vel"][sample].clone()
            q = self.spawn_states["quat"][sample].clone()
        tgt = torch.zeros(n, dtype=torch.long, device=dev)
        if self.demo is not None:
            m = use_demo
            k = torch.multinomial(
                self.demo_weights, n, replacement=True
            )
            p = torch.where(m[:, None], self.demo["pos"][k], p)
            v = torch.where(m[:, None], self.demo["vel"][k], v)
            q = torch.where(m[:, None], self.demo["quat"][k], q)
            tgt = torch.where(m, self.demo["gate"][k].long(), tgt)
            p = p + torch.randn(n, 3, device=dev) * cfg.start_noise_pos_m \
                * m[:, None]
            v = v + torch.randn(n, 3, device=dev) * cfg.start_noise_vel_mps \
                * m[:, None]
        q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
        self.p[idx] = p
        self.v[idx] = v
        self.q[idx] = q
        self.w[idx] = 0.0
        self.target[idx] = tgt
        self.t_ep[idx] = 0.0
        self.t_gate[idx] = 0.0
        self.prev_action[idx] = 0.0
        self.act_buf[:, idx] = 0.0
        self.act_delay[idx] = torch.randint(
            cfg.act_delay_steps_min, cfg.act_delay_steps_max + 1,
            (n,), device=dev,
        )
        if self.backbone is not None:
            if hasattr(self.backbone, "reset_nearest"):
                self.backbone.reset_nearest(idx, p)
            else:
                d = torch.linalg.norm(
                    self.backbone.P[None, :, :] - p[:, None, :], dim=-1
                )
                self.backbone.idx[idx] = d.argmin(dim=1)
                self.backbone.steps[idx] = self.backbone.idx[idx].clone()
        self.noise_pos[idx] = 0.0
        self.vis_age[idx] = 0.0
        self.vision_prev_fix[idx] = 0.0
        self.noise_amp[idx] = (
            cfg.pos_noise_lo
            + (cfg.pos_noise_hi - cfg.pos_noise_lo)
            * torch.rand(n, 1, device=dev)
        )
        self.progress[idx] = self._course_progress(p, tgt)
        self.best_prog[idx] = self.progress[idx]
        self.t_best[idx] = 0.0
        if self.demo is not None:
            self.spawn_flag[idx] = ~use_demo
        else:
            self.spawn_flag[idx] = True
        u = lambda lo, hi, *s: lo + (hi - lo) * torch.rand(*s, device=dev)
        self.reloc_next_t[idx] = u(*cfg.reloc_interval_s, n)
        self.reloc_end_t[idx] = 0.0
        self.reloc_rate[idx] = 0.0
        self.dr_thrust[idx] = u(*cfg.dr_thrust, n, 1)
        self.dr_K[idx] = u(*cfg.dr_rate_gain, n, 3)
        self.dr_tau[idx] = u(*cfg.dr_rate_tau, n, 3)
        self.dr_drag[idx] = u(*cfg.dr_drag, n, 3)
        if self.residual_ensemble is not None:
            member_count = getattr(
                self.residual_ensemble, "member_count", None
            )
            if member_count is None:
                member_count = len(self.residual_ensemble.members)
            self.residual_member[idx] = torch.randint(
                int(member_count), (n,), device=dev
            )
            self.world_model_disagreement[idx] = 0.0
            self.world_model_support_z[idx] = 0.0

    def _course_progress(self, p, tgt):
        tgt = torch.clamp(tgt, max=N_GATES - 1)
        start = self.track_points[tgt]
        end = self.track_points[tgt + 1]
        d = end - start
        L = self.seg_len[tgt]
        along = ((p - start) * d).sum(-1) / (L + 1e-9)
        along = torch.minimum(
            torch.clamp(along, min=0.0), L
        )
        return self.cum_len[tgt] + along

    def _corridor_dist(self, p):
        """(distance to track polyline, height below the local track line).

        dz > ~3 m means below the local floor anywhere on the course
        (gates sit 1.4-2 m above their floor) -- kills the underground
        loitering exploit without needing a terrain model.
        """
        a = self.track_points[:-1][None]      # (1,S,3)
        b = self.track_points[1:][None]
        d = b - a
        denom = (d * d).sum(-1)
        f = ((p[:, None] - a) * d).sum(-1) / (denom + 1e-9)
        f = torch.clamp(f, 0.0, 1.0)
        c = a + f[..., None] * d
        dist = torch.linalg.norm(c - p[:, None], dim=-1)
        k = dist.argmin(dim=1)
        rows = torch.arange(len(p), device=p.device)
        below = p[:, 2] - c[rows, k, 2]       # z down: positive = lower
        return dist[rows, k], below

    def observations(self) -> torch.Tensor:
        R = self._qmat(self.q)                    # body->world
        Rt = R.transpose(1, 2)
        p_noisy = self.p + self.noise_pos
        gi = torch.clamp(self.target, max=N_GATES - 1)
        rels, tans = [], []
        for la in range(N_LOOKAHEAD):
            gk = torch.clamp(gi + la, max=N_GATES - 1)
            rel = torch.einsum(
                "nij,nj->ni", Rt, self.gate_pos[gk] - p_noisy
            )
            rels.append(rel / 10.0)
            tans.append(torch.einsum(
                "nij,nj->ni", Rt, self.gate_tangent[gk]
            ))
        v_body = torch.einsum("nij,nj->ni", Rt, self.v)
        one_hot = torch.nn.functional.one_hot(gi, N_GATES).float()
        if self.cfg.fov_vision:
            # live realism (review find): the real policy sees the EKF's
            # REPORTED covariance, which grows during vision droughts and
            # resets on a fix -- it never sees the true error.  Model:
            # baseline sigma + measured coast growth (0.14 m/s) * age.
            sigma = torch.clamp(
                (0.09 + 0.14 * self.vis_age.unsqueeze(1)) / 0.5,
                0.0, 2.0,
            )
        else:
            sigma = torch.clamp(
                torch.linalg.norm(
                    self.noise_pos, dim=-1, keepdim=True
                ) / 0.5,
                0.0, 2.0,
            )
        conf = torch.cat([
            sigma, (gi.float() / (N_GATES - 1)).unsqueeze(1)
        ], dim=1)
        rates_obs = self.w / MAX_RATE
        obs = torch.cat([
            *rels, *tans, v_body / 10.0,
            R[:, :, 0], R[:, :, 1],
            rates_obs, self.prev_action, one_hot, conf,
        ], dim=1)
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _vision_fusion_probability(self) -> torch.Tensor:
        """Learned 10 Hz fusion probability from deployable state only."""
        if self.vision_outcome is None:
            return torch.full_like(self.vis_age, self.cfg.fov_detect_prob)
        rotation_t = self._qmat(self.q).transpose(1, 2)
        gi = torch.clamp(self.target, max=N_GATES - 1)
        rels = []
        for lookahead in range(3):
            gate = torch.clamp(gi + lookahead, max=N_GATES - 1)
            rels.append(torch.einsum(
                "nij,nj->ni",
                rotation_t,
                self.gate_pos[gate] - (self.p + self.noise_pos),
            ))
        rel = torch.stack(rels, dim=1)
        pitch = float(np.deg2rad(20.0))
        cos_pitch, sin_pitch = float(np.cos(pitch)), float(np.sin(pitch))
        depth = cos_pitch * rel[:, :, 0] - sin_pitch * rel[:, :, 2]
        safe_depth = torch.clamp(depth, min=0.25)
        horizontal = rel[:, :, 1].abs() / (
            safe_depth * float(np.tan(np.deg2rad(43.0)))
        )
        camera_down = sin_pitch * rel[:, :, 0] + cos_pitch * rel[:, :, 2]
        vertical = camera_down.abs() / (
            safe_depth * float(np.tan(np.deg2rad(27.0)))
        )
        speed = torch.linalg.norm(self.v, dim=1, keepdim=True)
        gate5 = torch.nn.functional.one_hot(
            torch.clamp(self.target, 0, 4), 5
        ).float()
        sigma_channel = torch.clamp(
            (0.09 + 0.14 * self.vis_age[:, None]) / 0.5,
            0.0,
            2.0,
        )
        feature = torch.cat([
            torch.clamp(torch.linalg.norm(rel, dim=2) / 30.0, 0.0, 3.0),
            torch.clamp(depth / 30.0, -1.0, 3.0),
            torch.clamp(horizontal, 0.0, 4.0),
            torch.clamp(vertical, 0.0, 4.0),
            torch.clamp(speed / 12.0, 0.0, 2.0),
            gate5,
            torch.clamp(self.vis_age[:, None] / 2.0, 0.0, 3.0),
            sigma_channel,
            self.vision_prev_fix[:, None],
        ], dim=1)
        normalized = (
            feature - self.vision_outcome["mean"]
        ) / self.vision_outcome["std"]
        return torch.sigmoid(
            normalized @ self.vision_outcome["weight"]
            + self.vision_outcome["bias"]
        )

    def _vision_uniform(self) -> torch.Tensor:
        worlds = int(self.cfg.common_random_worlds)
        if worlds > 0 and self.cfg.n_envs % worlds == 0:
            return torch.rand(worlds, device=self.device).repeat(
                self.cfg.n_envs // worlds
            )
        return torch.rand(self.cfg.n_envs, device=self.device)

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        cfg = self.cfg
        dev = self.device
        n = cfg.n_envs
        m = self.model
        if self.backbone is not None:
            if hasattr(self.backbone, "set_target"):
                self.backbone.set_target(self.target)
            if hasattr(self.backbone, "set_previous_action"):
                self.backbone.set_previous_action(self.prev_action)
            base = self.backbone.action(
                self.p + self.noise_pos, self.v, self._qmat(self.q)
            )
            residual = torch.clamp(action, -1.0, 1.0)
            if cfg.residual_active_gates:
                active = torch.zeros(
                    n, dtype=torch.bool, device=dev
                )
                for gate_index in cfg.residual_active_gates:
                    active |= self.target == int(gate_index)
                residual = residual * active[:, None]
            if cfg.residual_gate_scales:
                table = torch.as_tensor(
                    cfg.residual_gate_scales, dtype=torch.float32, device=dev
                )
                gate_row = torch.clamp(self.target, 0, len(table) - 1)
                residual = residual * table[gate_row, None]
            action = base + cfg.residual_scale * residual
        action = torch.clamp(action, -1.0, 1.0)
        # action transport delay
        self.act_buf = torch.roll(self.act_buf, 1, dims=0)
        self.act_buf[0] = action
        eff = self.act_buf[
            self.act_delay, torch.arange(n, device=dev)
        ]
        wire_rates = eff[:, :3] * torch.tensor(
            WIRE_RATE_LIMIT, device=dev
        )
        wire_thrust = torch.clamp(
            0.5 * (eff[:, 3] + 1.0), max=cfg.thrust_wire_cap
        )
        # mirror the live stack's launch assist: guaranteed minimum
        # thrust while separating from the pad on gate 0
        assist = (self.target == 0) & (self.t_ep < 0.55)
        wire_thrust = torch.where(
            assist, torch.clamp(wire_thrust, min=0.30), wire_thrust
        )
        # The live observation/log records the command after launch assist.
        # Feed that same command to the learned residual and previous-action
        # state; using the pre-assist -0.94 thrust made launch an artificial
        # out-of-distribution state even though every real log says -0.40.
        applied_action = action.clone()
        applied_action[:, 3] = torch.where(
            assist,
            torch.maximum(
                applied_action[:, 3],
                torch.full_like(applied_action[:, 3], -0.40),
            ),
            applied_action[:, 3],
        )
        residual_features_now = None
        residual_rotation = None
        if self.residual_ensemble is not None:
            from aigp.fastsim.worldmodel import residual_features
            residual_rotation = self._qmat(self.q)
            residual_features_now = residual_features(
                self.v, residual_rotation, self.w,
                applied_action, self.prev_action,
            )

        K = (
            cfg.rate_gain_sign
            * torch.tensor([abs(g) for g in m.rate_gain], device=dev)
            * self.dr_K
        )
        tau = torch.tensor(m.rate_tau, device=dev) * self.dr_tau
        thrust_acc = (
            (m.thrust_gain * wire_thrust
             + m.thrust_quad * wire_thrust ** 2)
            * self.dr_thrust[:, 0]
        )
        g_vec = torch.tensor([0.0, 0.0, 9.81], device=dev)
        drag_quad = torch.tensor(
            m.drag_quad if getattr(m, "drag_quad", None) else [0.0, 0.0, 0.0],
            device=dev,
        )
        dt = 1.0 / (cfg.control_hz * cfg.substeps)
        p_prev = self.p.clone()
        prev_plane = self._gate_local(self.p)

        for _ in range(cfg.substeps):
            self.w = self.w + dt * (K * wire_rates - self.w) / tau
            dq = self._qrotvec(self.w, dt)
            self.q = self._qmul(self.q, dq)
            self.q = self.q / torch.linalg.norm(
                self.q, dim=-1, keepdim=True
            )
            R = self._qmat(self.q)
            v_body = torch.einsum("nij,nj->ni", R.transpose(1, 2), self.v)
            f_body = -self.dr_drag * v_body
            f_body = f_body - drag_quad * torch.linalg.norm(
                self.v, dim=-1, keepdim=True
            ) * v_body
            f_body[:, 2] = f_body[:, 2] - thrust_acc
            a = g_vec + torch.einsum("nij,nj->ni", R, f_body)
            self.v = self.v + a * dt
            self.p = self.p + self.v * dt
            # spawn pad ground contact (z down; pad surface z=0 near
            # spawn): rest instead of falling through the world
            on_pad = (self.p[:, 2] > -0.02) & (self.p[:, 0] < 8.0) \
                & (self.p[:, 0].abs() < 12.0) & (self.p[:, 1].abs() < 8.0)
            if on_pad.any():
                self.p[:, 2] = torch.where(
                    on_pad, torch.full_like(self.p[:, 2], -0.02),
                    self.p[:, 2],
                )
                self.v[:, 2] = torch.where(
                    on_pad & (self.v[:, 2] > 0),
                    torch.zeros_like(self.v[:, 2]), self.v[:, 2],
                )
                self.v[:, :2] = torch.where(
                    on_pad[:, None], self.v[:, :2] * 0.7, self.v[:, :2]
                )

        if self.residual_ensemble is not None:
            from aigp.fastsim.worldmodel import rotation_exp
            means, log_stds = self.residual_ensemble(residual_features_now)
            member = self.residual_member
            row = torch.arange(n, device=dev)
            if hasattr(self.residual_ensemble, "support_z_by_member"):
                support = self.residual_ensemble.support_z_by_member(
                    residual_features_now
                )
                self.world_model_support_z = (
                    support.mean(0) if cfg.world_model_use_mean
                    else support[member, row]
                )
            else:
                normalized_features = (
                    residual_features_now - self.residual_ensemble.x_mean
                ) / self.residual_ensemble.x_std
                self.world_model_support_z = torch.sqrt(
                    normalized_features.square().mean(1)
                )
            correction = (
                means.mean(0) if cfg.world_model_use_mean
                else means[member, row]
            )
            if cfg.world_model_aleatoric_scale > 0.0:
                selected_log_std = (
                    torch.logsumexp(log_stds, dim=0) - np.log(len(log_stds))
                    if cfg.world_model_use_mean else log_stds[member, row]
                )
                correction = correction + cfg.world_model_aleatoric_scale * (
                    torch.exp(selected_log_std)
                    * torch.randn_like(correction)
                )
            correction = cfg.world_model_residual_scale * correction
            self.v = self.v + torch.einsum(
                "nij,nj->ni", residual_rotation, correction[:, :3]
            )
            # Apply the learned attitude delta in body coordinates after the
            # nominal integration, matching the residual training target.
            delta_q = self._qrotvec(correction[:, 3:6], 1.0)
            self.q = self._qmul(self.q, delta_q)
            self.q = self.q / torch.clamp(
                torch.linalg.norm(self.q, dim=1, keepdim=True), min=1e-8
            )
            self.w = self.w + correction[:, 6:9]
            self.world_model_disagreement = means.std(0).square().mean(1).sqrt()

        # Domain-randomized, one-shot velocity impulses.  Apply these to the
        # physical state (not the EKF belief) so a policy must actually recover
        # rather than merely ignore a noisy observation.  The launch is kept
        # undisturbed because the live pad assist is a separate mechanism.
        impulse = torch.zeros(n, dtype=torch.bool, device=dev)
        if cfg.impulse_rate_hz > 0.0:
            impulse = (
                (self.t_ep >= cfg.impulse_start_s)
                & (torch.rand(n, device=dev)
                   < cfg.impulse_rate_hz / cfg.control_hz)
            )
            if impulse.any():
                k = int(impulse.sum())
                direction = torch.randn(k, 3, device=dev)
                direction[:, 2] *= cfg.impulse_vertical_scale
                direction = torch.nn.functional.normalize(direction, dim=1)
                lo, hi = cfg.impulse_velocity_mps
                magnitude = lo + (hi - lo) * torch.rand(k, 1, device=dev)
                self.v[impulse] = self.v[impulse] + direction * magnitude

        step_dt = 1.0 / cfg.control_hz
        self.t_ep += step_dt
        self.t_gate += step_dt
        # estimator noise OU
        rho = float(np.exp(-step_dt / cfg.pos_noise_tau_s))
        self.noise_pos = (
            rho * self.noise_pos
            + np.sqrt(1 - rho * rho) * self.noise_amp
            * torch.randn(n, 3, device=dev)
        )
        vision_fix_probability = torch.zeros(n, device=dev)
        got_fix = torch.zeros(n, dtype=torch.bool, device=dev)
        if cfg.fov_vision:
            # Vision requires the gate in the camera frustum.  Flight 6/7
            # both died to policies that sprint pitched-forward: the real
            # detector then sees floor, the belief coasts, and the drone
            # arrives at the gate half a metre wrong.  Model fixes as
            # available only when a lookahead gate projects into the
            # tilted camera FOV; drift accumulates while blind; a fix
            # snaps the error back to the OU baseline.  The sigma obs
            # channel tracks |noise_pos|, so the policy can feel
            # blindness and learn to fly camera-first.
            Rt_fov = self._qmat(self.q).transpose(1, 2)
            cos_p = float(np.cos(np.deg2rad(cfg.fov_cam_pitch_deg)))
            sin_p = float(np.sin(np.deg2rad(cfg.fov_cam_pitch_deg)))
            cam_f = torch.tensor([cos_p, 0.0, -sin_p], device=dev)
            cam_r = torch.tensor([0.0, 1.0, 0.0], device=dev)
            cam_d = torch.tensor([sin_p, 0.0, cos_p], device=dev)
            tan_h = float(np.tan(np.deg2rad(cfg.fov_half_h_deg)))
            tan_v = float(np.tan(np.deg2rad(cfg.fov_half_v_deg)))
            visible = torch.zeros(n, dtype=torch.bool, device=dev)
            gi_now = torch.clamp(self.target, max=N_GATES - 1)
            lookahead_count = max(1, int(cfg.fov_gate_lookahead))
            gate_candidates = (
                [torch.full_like(gi_now, gate_index)
                 for gate_index in range(N_GATES)]
                if lookahead_count >= N_GATES
                else [
                    torch.clamp(gi_now + lookahead, max=N_GATES - 1)
                    for lookahead in range(lookahead_count)
                ]
            )
            for gk in gate_candidates:
                world = self.gate_pos[gk] - self.p
                rng = torch.linalg.norm(world, dim=-1)
                body = torch.einsum("nij,nj->ni", Rt_fov, world)
                zc = body @ cam_f
                xc = body @ cam_r
                yc = body @ cam_d
                in_fov = (
                    (zc > 0.3)
                    & (xc.abs() <= tan_h * zc)
                    & (yc.abs() <= tan_v * zc)
                    & (rng < cfg.fov_max_range_m)
                )
                visible = visible | in_fov
            self.vis_age = self.vis_age + step_dt
            due = self.vis_age >= 1.0 / cfg.fov_fix_rate_hz
            vision_fix_probability = self._vision_fusion_probability()
            got_fix = visible & due & (
                self._vision_uniform() < vision_fix_probability
            )
            coasting = self.vis_age > cfg.fov_coast_after_s
            spd_now = torch.linalg.norm(self.v, dim=-1)
            self.noise_pos = self.noise_pos + coasting.float()[:, None] * (
                cfg.coast_speed_diffuse * spd_now[:, None] * step_dt
                * torch.randn(n, 3, device=dev)
                + cfg.coast_speed_bias * spd_now[:, None] * step_dt
                * torch.nn.functional.normalize(
                    self.noise_pos + 1e-6 * torch.randn(
                        n, 3, device=dev
                    ), dim=-1,
                )
            )
            if got_fix.any():
                k = int(got_fix.sum())
                self.noise_pos[got_fix] = (
                    self.noise_amp[got_fix]
                    * torch.randn(k, 3, device=dev)
                )
                self.vis_age[got_fix] = 0.0
            self.vision_prev_fix = got_fix.float()
        if cfg.reloc_events:
            # MEASURED live behavior (flight round 4 forensics): the
            # filter coasts ~0.5-1.5s on IMU during FAST gate approaches
            # (2 Hz vision cannot refix in time), drifting ~0.05 m per
            # m/s of speed per second. Model: continuous coast drift
            # proportional to speed, PLUS the discrete reloc events.
            # (When fov_vision is on, the FOV block owns coast drift.)
            spd_now = torch.linalg.norm(self.v, dim=-1)
            if not cfg.fov_vision:
                self.noise_pos = self.noise_pos + (
                    cfg.coast_speed_diffuse * spd_now[:, None] * step_dt
                    * torch.randn(n, 3, device=dev)
                    + cfg.coast_speed_bias * spd_now[:, None] * step_dt
                    * torch.nn.functional.normalize(
                        self.noise_pos + 1e-6 * torch.randn(
                            n, 3, device=dev
                        ), dim=-1,
                    )
                )
            start = self.t_ep >= self.reloc_next_t
            if start.any():
                k = int(start.sum())
                dur = (cfg.reloc_drift_s[0]
                       + (cfg.reloc_drift_s[1] - cfg.reloc_drift_s[0])
                       * torch.rand(k, device=dev))
                mag = (cfg.reloc_mag_m[0]
                       + (cfg.reloc_mag_m[1] - cfg.reloc_mag_m[0])
                       * torch.rand(k, device=dev))
                d = torch.randn(k, 3, device=dev)
                d = d / (torch.linalg.norm(d, dim=1, keepdim=True) + 1e-9)
                self.reloc_dir[start] = d
                self.reloc_rate[start] = mag / dur
                self.reloc_end_t[start] = self.t_ep[start] + dur
                self.reloc_next_t[start] = (
                    self.t_ep[start] + dur
                    + cfg.reloc_interval_s[0]
                    + (cfg.reloc_interval_s[1] - cfg.reloc_interval_s[0])
                    * torch.rand(k, device=dev)
                )
            drifting = self.t_ep < self.reloc_end_t
            self.noise_pos = self.noise_pos + (
                drifting.float()[:, None]
                * self.reloc_dir * self.reloc_rate[:, None] * step_dt
            )
            # the snap: drift just ended -> collapse back to baseline
            snapped = (~drifting) & (self.reloc_end_t > 0) \
                & (self.t_ep - step_dt < self.reloc_end_t)
            if snapped.any():
                self.noise_pos[snapped] = (
                    self.noise_amp[snapped]
                    * torch.randn(int(snapped.sum()), 3, device=dev)
                )

        # gate plane events
        new_plane = self._gate_local(self.p)
        crossed = (prev_plane[:, 1] < 0.0) & (new_plane[:, 1] >= 0.0)
        frac = torch.where(
            crossed,
            -prev_plane[:, 1]
            / (new_plane[:, 1] - prev_plane[:, 1] + 1e-9),
            torch.zeros_like(new_plane[:, 1]),
        )
        cross_local = prev_plane + frac[:, None] * (new_plane - prev_plane)
        r_inf = torch.maximum(
            cross_local[:, 0].abs(), cross_local[:, 2].abs()
        )
        passed = crossed & (r_inf < HOLE_HALF)
        hit = crossed & (r_inf >= HOLE_HALF) & (r_inf < PANEL_HALF + 0.10)

        progress = self._course_progress(self.p, self.target)
        d_prog = torch.clamp(progress - self.progress, -0.5, 1.2)
        reward = (
            cfg.progress_scale * d_prog
            - cfg.time_penalty_per_s * step_dt
            - cfg.action_smoothness
            * ((applied_action - self.prev_action) ** 2).sum(-1)
        )
        clearance = torch.clamp(
            (HOLE_HALF - r_inf) / HOLE_HALF, 0.0, 1.0
        )
        reward = reward + passed.float() * (
            cfg.gate_bonus + cfg.clearance_bonus * clearance
        )
        gate_before_pass = torch.clamp(self.target, max=N_GATES - 1)
        if cfg.gate_time_targets_s and cfg.gate_time_bonus_per_s != 0.0:
            segment_seconds_saved = torch.clamp(
                self.gate_time_targets[gate_before_pass] - self.t_gate,
                min=-1.0,
                max=1.0,
            )
            reward = reward + (
                passed.float()
                * cfg.gate_time_bonus_per_s
                * segment_seconds_saved
            )
        demo_cross_error = torch.linalg.norm(
            cross_local[:, [0, 2]] - self.demo_cross_local[gate_before_pass],
            dim=1,
        )
        demo_cross_score = torch.clamp(
            1.0 - demo_cross_error
            / max(float(cfg.demo_crossing_radius_m), 1e-6),
            0.0,
            1.0,
        )
        reward = reward + (
            passed.float()
            * self.demo_cross_valid[gate_before_pass].float()
            * cfg.demo_crossing_bonus
            * demo_cross_score
        )
        self.target = torch.where(
            passed, self.target + 1, self.target
        )
        self.t_gate = torch.where(
            passed, torch.zeros_like(self.t_gate), self.t_gate
        )
        finished = self.target >= cfg.race_gates
        reward = reward + finished.float() * cfg.finish_bonus
        if (
            cfg.finish_time_target_s > 0.0
            and cfg.finish_time_bonus_per_s != 0.0
        ):
            seconds_saved = torch.clamp(
                cfg.finish_time_target_s - self.t_ep, min=-2.0, max=4.0
            )
            reward = reward + (
                finished.float()
                * self.spawn_flag.float()
                * cfg.finish_time_bonus_per_s
                * seconds_saved
            )

        corridor, below = self._corridor_dist(self.p)
        off = (corridor > cfg.corridor_m) | (below > 3.0)
        demo_dist = torch.zeros_like(corridor)
        if self.demo_path is not None:
            f = ((self.p[:, None] - self.demo_path_a[None])
                 * self.demo_path_d[None]).sum(-1) / self.demo_path_len2
            f = torch.clamp(f, 0.0, 1.0)
            closest = self.demo_path_a[None] \
                + f[..., None] * self.demo_path_d[None]
            demo_dist = torch.linalg.norm(
                closest - self.p[:, None], dim=-1
            ).min(dim=1).values
            off = off | (demo_dist > cfg.demo_corridor_m)
            demo_excess = torch.clamp(
                demo_dist - cfg.demo_tracking_free_m, min=0.0
            )
            reward = reward - (
                cfg.demo_tracking_penalty_per_s
                * demo_excess.square()
                * step_dt
            )
        overspeed = torch.linalg.norm(self.v, dim=-1) > cfg.speed_cap_mps
        gate_limit = torch.where(
            self.target == 0,
            torch.full_like(self.t_gate, cfg.live_gate_timeout_s + 3.0),
            torch.full_like(self.t_gate, cfg.live_gate_timeout_s),
        )
        timeout_gate = self.t_gate > gate_limit
        # live tilt guard: body z vs world z (z down, upright ~ +1)
        R_now = self._qmat(self.q)
        tilt_cos = R_now[:, 2, 2]
        inverted = tilt_cos < float(
            np.cos(np.deg2rad(cfg.max_tilt_deg))
        )
        timeout_gate = timeout_gate | inverted
        # live no-progress guard
        improved = progress > self.best_prog + 0.05
        self.best_prog = torch.where(improved, progress, self.best_prog)
        self.t_best = torch.where(
            improved, self.t_ep.clone(), self.t_best
        )
        timeout_gate = timeout_gate | (
            (self.t_ep - self.t_best) > cfg.no_progress_s
        )
        timeout_ep = self.t_ep > cfg.max_episode_s
        if self.obstacles is not None:
            dxy = torch.linalg.norm(
                self.p[:, None, :2] - self.obs_xy[None], dim=-1
            )
            in_z = (self.p[:, None, 2] > (self.obs_z_root - self.obs_h)[None]) \
                & (self.p[:, None, 2] < (self.obs_z_root + 0.5)[None])
            ob_hit = ((dxy < self.obs_r[None]) & in_z).any(dim=1)
            hit = hit | ob_hit
        reward = reward - hit.float() * cfg.collision_penalty
        # every non-finish terminal costs the same order as crashing --
        # otherwise sitting on the pad (time penalty only) is the
        # rational policy and launch is never learned
        reward = reward - (
            off | overspeed | timeout_gate
        ).float() * cfg.offtrack_penalty

        terminated = hit | off | overspeed | timeout_gate | finished
        truncated = timeout_ep & ~terminated
        done = terminated | truncated

        self.progress = torch.where(
            passed, self._course_progress(self.p, self.target), progress
        )
        self.prev_action = applied_action.clone()
        info = {
            "passed": passed,
            "hit": hit,
            # hole-plane crossing offset (inf-norm, m) for clearance
            # audits; -1 when the step crossed no gate plane
            "cross_r": torch.where(
                crossed, r_inf, torch.full_like(r_inf, -1.0)
            ),
            # Preserve signed crossing coordinates for model/live parity
            # diagnostics.  Clearance alone hides systematic side bias.
            "cross_lateral_m": torch.where(
                crossed,
                cross_local[:, 0],
                torch.full_like(cross_local[:, 0], float("nan")),
            ),
            "cross_vertical_m": torch.where(
                crossed,
                cross_local[:, 2],
                torch.full_like(cross_local[:, 2], float("nan")),
            ),
            "demo_dist": demo_dist,
            "demo_cross_error": torch.where(
                crossed,
                demo_cross_error,
                torch.full_like(demo_cross_error, -1.0),
            ),
            "cross_gate": torch.where(
                crossed,
                gate_before_pass,
                torch.full_like(gate_before_pass, -1),
            ),
            "finished": finished,
            "off": off,
            "overspeed": overspeed,
            "timeout": timeout_gate,
            "speed": torch.linalg.norm(self.v, dim=-1),
            "world_model_disagreement": self.world_model_disagreement.clone(),
            "world_model_support_z": self.world_model_support_z.clone(),
            "t_ep": self.t_ep.clone(),
            "target": self.target.clone(),
            "spawn_done": done & self.spawn_flag,
            "spawn_launched": done & self.spawn_flag & (self.target > 0),
            "impulse": impulse,
            "vision_fix": got_fix,
            "vision_fix_probability": vision_fix_probability,
        }
        idx = torch.nonzero(done).squeeze(-1)
        if len(idx):
            self.reset(idx)
        return self.observations(), reward, done, info

    def _gate_local(self, p):
        gi = torch.clamp(self.target, max=N_GATES - 1)
        rel = p - self.gate_pos[gi]
        return torch.einsum(
            "nij,ni->nj", self.gate_R[gi], rel
        )  # columns: x,y(normal-ish),z in gate frame
