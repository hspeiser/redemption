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
    control_hz: float = 30.0
    substeps: int = 4
    max_episode_s: float = 45.0
    # reward
    progress_scale: float = 2.0
    gate_bonus: float = 25.0
    finish_bonus: float = 600.0
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
    rate_gain_sign: float = 1.0        # pinned by attitude check
    thrust_wire_cap: float = 0.52      # live stack caps wire thrust
    random_start_frac: float = 0.6
    start_noise_pos_m: float = 0.4
    start_noise_vel_mps: float = 0.8
    speed_cap_mps: float = 16.0

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


class FastVQ2Env:
    def __init__(
        self,
        model: SurrogateModel,
        map_path: str | Path,
        demo_states: dict | None = None,
        config: FastEnvConfig | None = None,
        device: str = "cuda",
        obstacles_path: str | Path | None = None,
    ) -> None:
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

        z = lambda *shape: torch.zeros(*shape, device=self.device)
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
        if self.demo is not None:
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
            0, cfg.act_delay_steps_max + 1, (n,), device=dev
        )
        self.noise_pos[idx] = 0.0
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
        sigma = torch.clamp(
            torch.linalg.norm(self.noise_pos, dim=-1, keepdim=True) / 0.5,
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

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        cfg = self.cfg
        dev = self.device
        n = cfg.n_envs
        m = self.model
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
        if cfg.reloc_events:
            # MEASURED live behavior (flight round 4 forensics): the
            # filter coasts ~0.5-1.5s on IMU during FAST gate approaches
            # (2 Hz vision cannot refix in time), drifting ~0.05 m per
            # m/s of speed per second. Model: continuous coast drift
            # proportional to speed, PLUS the discrete reloc events.
            spd_now = torch.linalg.norm(self.v, dim=-1)
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
            * ((action - self.prev_action) ** 2).sum(-1)
        )
        clearance = torch.clamp(
            (HOLE_HALF - r_inf) / HOLE_HALF, 0.0, 1.0
        )
        reward = reward + passed.float() * (
            cfg.gate_bonus + cfg.clearance_bonus * clearance
        )
        self.target = torch.where(
            passed, self.target + 1, self.target
        )
        self.t_gate = torch.where(
            passed, torch.zeros_like(self.t_gate), self.t_gate
        )
        finished = self.target >= N_GATES
        reward = reward + finished.float() * cfg.finish_bonus

        corridor, below = self._corridor_dist(self.p)
        off = (corridor > cfg.corridor_m) | (below > 3.0)
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
        self.prev_action = action.clone()
        info = {
            "passed": passed,
            "hit": hit,
            "finished": finished,
            "off": off,
            "overspeed": overspeed,
            "timeout": timeout_gate,
            "speed": torch.linalg.norm(self.v, dim=-1),
            "t_ep": self.t_ep.clone(),
            "target": self.target.clone(),
            "spawn_done": done & self.spawn_flag,
            "spawn_launched": done & self.spawn_flag & (self.target > 0),
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
