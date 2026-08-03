"""Batched port of the deployed VQ2 live-teacher composition.

Deployment-parity evaluator for line/schedule searches: replaces the
generic FlatRefController with a faithful batched implementation of the
controller that actually flies (train_vq2_sac_live.VQ2SACLearner.action
in its deterministic candidate-arm configuration), driven by a frozen
session config.  Scope = the components ACTIVE in the 35.374s champion
config: nearest-row reference with per-gate windows/leads, per-gate
gains/biases/limits, gate-center funnel, reference lateral/vertical
offset interpolation, per-gate thrust/rate/velocity scales, trajectory
blend through aigp.flight.RateController semantics (incl. the z
integrator and the legacy 40u thrust mapping -- deliberately preserved),
teacher_blend, and gate-routed primary/secondary PPO residuals with
their own observation normalizers.

Inactive-by-config components (sequential mode, predictive handoff,
frozen profiles, schedules, macro exploration, line backbone) are NOT
ported; the constructor refuses configs that enable them so parity can
never silently lie.

Parity contract: scripts/liveteacher_parity.py layer 1 must reproduce
the recorded 35.37 lap's selected_reference_row / teacher_action /
final action before any rollout statistics are trusted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from aigp.flight import RATE_CMD_SIGN
from aigp.rl.vq2_features import WIRE_RATE_LIMIT
from aigp.rl.sac import GaussianActor

ACT_DIM = 4
OBS_DIM = 53
N_RACE_GATES = 17
G = 9.81
DT = 1.0 / 30.0


def _parse_map(text, cast=float):
    out = {}
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, value = part.split(":")
        out[int(key)] = cast(value)
    return out


def _parse_set(text):
    return {int(v) for v in str(text or "").split(",") if v.strip()}


def observation_rotation_batch(obs: np.ndarray) -> np.ndarray:
    """Vectorized copy of train_vq2_sac_live.observation_rotation."""
    first = obs[:, 21:24].astype(float).copy()
    second = obs[:, 24:27].astype(float).copy()
    first /= np.linalg.norm(first, axis=1, keepdims=True) + 1e-9
    second -= first * np.sum(first * second, axis=1, keepdims=True)
    second /= np.linalg.norm(second, axis=1, keepdims=True) + 1e-9
    third = np.cross(first, second)
    return np.stack([first, second, third], axis=2)


def estimate_demo_acceleration(velocity, wall, source_episode):
    """Verbatim port of the trainer's smooth per-lap acceleration."""
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
        window = min(
            15, len(indices) if len(indices) % 2 else len(indices) - 1)
        if window >= 5:
            smooth_v = savgol_filter(
                v, window_length=window,
                polyorder=min(3, window - 2), axis=0, mode="interp")
            smooth_a = savgol_filter(
                smooth_v, window_length=window,
                polyorder=min(3, window - 2), deriv=1, delta=dt,
                axis=0, mode="interp")
        else:
            smooth_a = np.gradient(v, dt, axis=0)
        magnitude = np.linalg.norm(smooth_a, axis=1)
        scale = np.minimum(1.0, 18.0 / np.maximum(magnitude, 1e-6))
        acceleration[indices] = smooth_a * scale[:, None]
    return acceleration.astype(np.float32)


def _rotvec_batch(E: torch.Tensor) -> torch.Tensor:
    """Batched SO(3) log map (same construction as lineopt)."""
    trace = E[:, 0, 0] + E[:, 1, 1] + E[:, 2, 2]
    ang = torch.arccos(torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0))
    vee = torch.stack([E[:, 2, 1] - E[:, 1, 2],
                       E[:, 0, 2] - E[:, 2, 0],
                       E[:, 1, 0] - E[:, 0, 1]], dim=1)
    sin_a = torch.sin(ang).clamp(min=1e-6)
    return vee * (ang / (2.0 * sin_a))[:, None]


class BatchedLiveTeacher:
    """Backbone-compatible batched deployment controller.

    FastVQ2Env calls action(p, v, R) plus, when needs_extras is set,
    prev_action and target -- both required by the nearest-row metric
    and gate routing.
    """

    needs_extras = True

    def __init__(self, config_path, n_envs, device="cuda",
                 map_path=None, demo_path=None):
        cfg = json.loads(Path(config_path).read_text())["args"]
        self.cfg = cfg
        dev = torch.device(device)
        self.dev = dev
        self.n_envs = n_envs

        # refuse configs whose active components exceed the ported scope
        assert cfg.get("reference_mode", "nearest") == "nearest"
        assert not _parse_map(cfg.get("reference_sequential_speeds", ""))
        assert not _parse_map(cfg.get("predictive_handoff_distances", ""))
        assert cfg.get("frozen_residual_episode") is None
        assert cfg.get("frozen_action_episode") is None
        assert cfg.get("ppo_residual_schedule") is None
        assert cfg.get("line") is None
        assert not _parse_map(cfg.get("teacher_blends", ""))
        assert float(cfg.get("teacher_blend", 1.0)) == 1.0, \
            "SAC teacher actor path not ported; needs teacher_blend 1.0"
        assert not _parse_map(cfg.get("actor_speed_limits", ""))
        assert not _parse_map(cfg.get("residual_phase_windows", ""))
        assert not _parse_map(cfg.get("longitudinal_position_gains", ""))
        assert not _parse_map(cfg.get("longitudinal_velocity_gains", ""))
        assert float(cfg.get("longitudinal_position_gain", 0.0)) == 0.0
        assert float(cfg.get("longitudinal_velocity_gain", 0.0)) == 0.0
        assert not _parse_map(cfg.get("lateral_feedback_limits", ""))
        assert not _parse_set(cfg.get("trajectory_blend_gates", ""))
        assert not _parse_map(cfg.get("residual_scale_gates", ""))

        map_file = Path(map_path or cfg["map"])
        gate_map = json.loads(map_file.read_text())["gates"]
        self.gate_pos_map = np.asarray(
            [g["pos"] for g in gate_map[:17]], np.float32)
        quat = np.asarray([g["quat_wxyz"] for g in gate_map[:17]], float)
        gate_rot = Rotation.from_quat(np.stack(
            [quat[:, 1], quat[:, 2], quat[:, 3], quat[:, 0]], axis=1
        )).as_matrix()
        ref_rot = gate_rot.copy()
        for gi in range(17):
            incoming = self.gate_pos_map[gi] - (
                self.gate_pos_map[gi - 1] if gi else np.zeros(3))
            if np.dot(ref_rot[gi, :, 1], incoming) < 0.0:
                ref_rot[gi, :, 0] *= -1.0
                ref_rot[gi, :, 1] *= -1.0
        lat_off = _parse_map(cfg.get("reference_lateral_offsets", ""))
        vert_off = _parse_map(cfg.get("reference_vertical_offsets", ""))
        offsets = np.zeros((17, 3), np.float32)
        for gi in range(17):
            offsets[gi] = (lat_off.get(gi, 0.0) * ref_rot[gi, :, 0]
                           + vert_off.get(gi, 0.0) * ref_rot[gi, :, 2])
        self.gate_offsets = torch.tensor(offsets, device=dev)
        self.gate_normal = torch.tensor(
            gate_rot[:, :, 1], dtype=torch.float32, device=dev)

        # demo (the hybrid reference) exactly as the trainer loads it
        demo = np.load(Path(demo_path or cfg["demo"]), allow_pickle=False)
        d_obs = np.asarray(demo["observation"], np.float32)
        self.demo_obs = torch.tensor(d_obs, device=dev)
        self.demo_action = torch.tensor(
            np.asarray(demo["action"], np.float32), device=dev)
        d_vel = np.asarray(demo["velocity"], np.float32)
        self.demo_vel = torch.tensor(d_vel, device=dev)
        self.demo_acc = torch.tensor(estimate_demo_acceleration(
            d_vel, np.asarray(demo["wall"], np.float64),
            demo["source_episode"] if "source_episode" in demo.files
            else None), device=dev)
        d_gate = np.asarray(demo["gate_index"], np.int64)
        self.demo_gate = torch.tensor(d_gate, device=dev)
        d_pos = np.asarray(demo["position"], np.float32)
        rot = observation_rotation_batch(d_obs)
        gate_vec_world = np.einsum("nij,nj->ni", rot, d_obs[:, :3]) * 10.0
        gate_position = np.zeros((17, 3), np.float32)
        for gi in range(17):
            mask = d_gate == gi
            gate_position[gi] = np.median(
                d_pos[mask] + gate_vec_world[mask], axis=0)
        self.demo_gate_position = torch.tensor(gate_position, device=dev)
        self.demo_gate_vec_world = torch.tensor(
            gate_vec_world.astype(np.float32), device=dev)
        # demo reference rotations precomputed per row
        self.demo_rot = torch.tensor(
            rot.astype(np.float32), device=dev)
        seg_start = np.zeros(17, np.int64)
        seg_end = np.zeros(17, np.int64)
        for gi in range(17):
            rows = np.flatnonzero(d_gate == gi)
            if not len(rows):
                raise ValueError(f"demo has no rows for gate {gi}")
            seg_start[gi] = rows[0]
            seg_end[gi] = rows[-1]
        self.seg_start = torch.tensor(seg_start, device=dev)
        self.seg_end = torch.tensor(seg_end, device=dev)

        # per-gate scalar tables
        def table(default, mapping, special_gate=None, special_scale=1.0):
            t = np.full(17, float(default), np.float32)
            if special_gate is not None and 0 <= special_gate < 17:
                t[special_gate] = default * special_scale
            for k, v in mapping.items():
                t[k] = v
            return torch.tensor(t, device=dev)

        self.lat_gain_scale = table(
            1.0, _parse_map(cfg.get("lateral_gain_scales", "")),
            int(cfg.get("special_lateral_gate", -1)),
            float(cfg.get("special_lateral_gain_scale", 1.0)))
        self.vert_gain_scale = table(1.0, {},
                                     int(cfg.get("special_vertical_gate",
                                                 -1)),
                                     float(cfg.get(
                                         "special_vertical_gain_scale",
                                         1.0)))
        lat_bias = np.zeros(17, np.float32)
        for gi in _parse_set(cfg.get("lateral_bias_gates", "")):
            lat_bias[gi] += float(cfg.get("lateral_action_bias", 0.0))
        for gi in _parse_set(cfg.get("right_lateral_bias_gates", "")):
            lat_bias[gi] += float(cfg.get("right_lateral_action_bias",
                                          0.0))
        for gi, v in _parse_map(cfg.get("extra_lateral_biases",
                                        "")).items():
            lat_bias[gi] += v
        self.lat_bias = torch.tensor(lat_bias, device=dev)
        vert_bias = np.zeros(17, np.float32)
        for gi in _parse_set(cfg.get("vertical_bias_gates", "")):
            vert_bias[gi] += float(cfg.get("vertical_action_bias", 0.0))
        for gi, v in _parse_map(cfg.get("extra_vertical_biases",
                                        "")).items():
            vert_bias[gi] += v
        self.vert_bias = torch.tensor(vert_bias, device=dev)
        self.lat_pos_gain = float(cfg.get("lateral_position_gain", 0.0))
        self.lat_vel_gain = float(cfg.get("lateral_velocity_gain", 0.0))
        self.vert_pos_gain = float(cfg.get("vertical_position_gain", 0.0))
        self.vert_vel_gain = float(cfg.get("vertical_velocity_gain", 0.0))
        self.thrust_scale = table(
            float(cfg.get("reference_thrust_scale", 1.0)),
            _parse_map(cfg.get("reference_thrust_scales", "")))
        self.rate_scale = table(
            float(cfg.get("reference_rate_scale", 1.0)),
            _parse_map(cfg.get("reference_rate_scales", "")))
        self.vel_scale = table(
            float(cfg.get("reference_velocity_scale", 1.0)),
            _parse_map(cfg.get("reference_velocity_scales", "")))
        self.action_lead = table(
            int(cfg.get("reference_action_lead", 0)),
            _parse_map(cfg.get("reference_action_leads", ""), cast=int)
        ).long()
        self.traj_blend = table(
            float(cfg.get("trajectory_blend", 0.0)),
            _parse_map(cfg.get("trajectory_blends", "")))
        self.max_advance = int(cfg.get("reference_max_advance", 8))
        self.max_retreat = int(cfg.get("reference_max_retreat", 2))
        self.feedback_scale = float(
            cfg.get("reference_feedback_scale", 1.0))
        self.funnel_gates = _parse_set(
            cfg.get("gate_center_funnel_gates", ""))
        funnel_mask = np.zeros(17, np.float32)
        for gi in (self.funnel_gates or range(17)):
            funnel_mask[gi] = 1.0
        self.funnel_mask = torch.tensor(funnel_mask, device=dev)
        self.funnel_dist = float(
            cfg.get("gate_center_funnel_distance", 0.0))
        self.funnel_full = float(
            cfg.get("gate_center_funnel_full_distance", 0.0))
        self.funnel_strength = float(np.clip(
            cfg.get("gate_center_funnel_strength", 0.0), 0.0, 1.0))

        # trajectory controller constants (aigp.flight.RateController with
        # the trainer's overrides)
        kp_scale = float(cfg.get("trajectory_kp_scale", 1.0))
        kv_scale = float(cfg.get("trajectory_kv_scale", 1.0))
        self.traj_kp = torch.tensor(
            [1.2, 1.2, 2.0], device=dev) * max(kp_scale, 0.0)
        self.traj_kv = torch.tensor(
            [2.0, 2.0, 2.8], device=dev) * max(kv_scale, 0.0)
        self.traj_katt = max(
            float(cfg.get("trajectory_attitude_gain", 4.0)), 0.0)
        self.traj_rate_limit = 4.0
        self.traj_k_thrust = 40.0
        self.traj_thrust_limit = 0.52
        from aigp.fastsim.sysid import SurrogateModel
        line_model = SurrogateModel.load(str(cfg["line_model"]))
        self.traj_rate_cmd_gain = torch.tensor(
            1.0 / np.maximum(np.abs(np.asarray(
                line_model.rate_gain, np.float32)), 1e-3), device=dev)
        self.rate_cmd_sign = torch.tensor(
            np.asarray(RATE_CMD_SIGN, np.float32), device=dev)
        self.wire_rate_limit = torch.tensor(
            np.asarray(WIRE_RATE_LIMIT, np.float32), device=dev)

        # residual actors + routing
        self.residual_gates = _parse_set(cfg.get("residual_gates", ""))
        res_mask = np.zeros(17, np.float32)
        for gi in (self.residual_gates or range(17)):
            res_mask[gi] = 1.0
        self.residual_mask = torch.tensor(res_mask, device=dev)
        self.residual_scale = float(cfg.get("residual_scale", 0.0))
        self.residual_clip = float(cfg.get("residual_output_clip", 1.0))

        def load_actor(path):
            payload = torch.load(path, map_location="cpu",
                                 weights_only=False)
            actor = GaussianActor(OBS_DIM, ACT_DIM).to(dev).eval()
            actor.load_state_dict(payload["actor"])
            for p in actor.parameters():
                p.requires_grad_(False)
            mean = torch.tensor(np.asarray(
                payload["obs_mean"].detach().cpu(), np.float32),
                device=dev)
            std = torch.tensor(np.sqrt(np.asarray(
                payload["obs_var"].detach().cpu(), np.float32) + 1e-6),
                device=dev)
            return actor, mean, std

        self.primary = (load_actor(cfg["ppo_residual_checkpoint"])
                        if cfg.get("ppo_residual_checkpoint") else None)
        self.secondary = (
            load_actor(cfg["secondary_ppo_residual_checkpoint"])
            if cfg.get("secondary_ppo_residual_checkpoint") else None)
        sec_gates = _parse_set(cfg.get("secondary_ppo_residual_gates",
                                       ""))
        sec_mask = np.zeros(17, np.float32)
        for gi in sec_gates:
            sec_mask[gi] = 1.0
        self.secondary_mask = torch.tensor(sec_mask, device=dev)

        # per-env state
        self.cursor = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.ref_gate = torch.full((n_envs,), -1, dtype=torch.long,
                                   device=dev)
        self.traj_zi = torch.zeros(n_envs, device=dev)
        # RefController-compat attributes for env reset hooks
        self.idx = self.cursor
        self.steps = torch.zeros(n_envs, dtype=torch.long, device=dev)

    def reset(self, env_ids):
        self.ref_gate[env_ids] = -1
        self.cursor[env_ids] = 0
        self.traj_zi[env_ids] = 0.0

    def reset_nearest(self, env_ids, p):
        self.reset(env_ids)

    @torch.no_grad()
    def action(self, p, v, R, prev_action=None, target=None,
               debug=False, obs_override=None):
        n = p.shape[0]
        dev = self.dev
        if prev_action is None:
            prev_action = torch.zeros(n, ACT_DIM, device=dev)
        if target is None:
            target = torch.zeros(n, dtype=torch.long, device=dev)
        g = torch.clamp(target, 0, N_RACE_GATES - 1)

        gate_vec_world = self.demo_gate_position[g] - p
        # gate-change cursor reset
        changed = self.ref_gate != g
        self.cursor = torch.where(changed, self.seg_start[g], self.cursor)
        self.ref_gate = g.clone()

        # candidate window
        offs = torch.arange(-self.max_retreat, self.max_advance + 1,
                            device=dev)
        cand = torch.clamp(self.cursor[:, None] + offs[None, :],
                           self.seg_start[g][:, None],
                           self.seg_end[g][:, None])          # (n,W)
        gate_start = self.seg_start[g].float()
        seg_rows = torch.clamp(
            (self.seg_end[g] - self.seg_start[g]).float(), min=1.0)
        phase = (cand.float() - gate_start[:, None]) / seg_rows[:, None]
        prev_off = torch.where(
            (g > 0)[:, None], self.gate_offsets[torch.clamp(g - 1, 0)],
            torch.zeros(n, 3, device=dev))
        cur_off = self.gate_offsets[g]
        geo_off = prev_off[:, None, :] + phase[..., None] * (
            cur_off - prev_off)[:, None, :]                    # (n,W,3)
        cand_vec = self.demo_gate_vec_world[cand] - geo_off
        vec_err = cand_vec - gate_vec_world[:, None, :]
        pa_err = ((self.demo_obs[cand][:, :, 30:34]
                   - prev_action[:, None, :]) ** 2).sum(-1)
        speed_err = (torch.linalg.norm(
            self.demo_obs[cand][:, :, 18:21], dim=-1)
            - (torch.linalg.norm(v, dim=-1) / 10.0)[:, None]) ** 2
        dist = (4.0 * (vec_err ** 2).sum(-1) + 2.0 * pa_err
                + 0.5 * speed_err)
        # de-duplicate clamped rows: keep first occurrence by inflating
        # duplicates (matches the trainer, whose candidate list is unique)
        dup = torch.zeros_like(dist, dtype=torch.bool)
        dup[:, 1:] = cand[:, 1:] == cand[:, :-1]
        dist = torch.where(dup, torch.full_like(dist, 1e9), dist)

        best = dist.argmin(dim=1)
        self.cursor = cand[torch.arange(n, device=dev), best]
        k4 = torch.topk(-dist, k=4, dim=1).indices            # (n,4)
        rows4 = torch.gather(cand, 1, k4)
        d4 = torch.gather(dist, 1, k4)
        w4 = 1.0 / torch.clamp(d4, min=1e-4)
        w4 = w4 / w4.sum(dim=1, keepdim=True)

        lead = self.action_lead[g]
        action_rows = torch.clamp(rows4 + lead[:, None],
                                  self.seg_start[g][:, None],
                                  self.seg_end[g][:, None])
        reference = (self.demo_action[action_rows]
                     * w4[..., None]).sum(dim=1)               # (n,4)
        rs = self.rate_scale[g]
        reference = torch.cat([
            torch.clamp(rs[:, None] * reference[:, :3], -1.0, 1.0),
            reference[:, 3:4]], dim=1)

        ref_obs = (self.demo_obs[rows4] * w4[..., None]).sum(dim=1)
        # reference rotation: weighted rows -> orthonormalized
        first = ref_obs[:, 21:24]
        first = first / (torch.linalg.norm(first, dim=1,
                                           keepdim=True) + 1e-9)
        second = ref_obs[:, 24:27]
        second = second - first * (first * second).sum(1, keepdim=True)
        second = second / (torch.linalg.norm(second, dim=1,
                                             keepdim=True) + 1e-9)
        third = torch.linalg.cross(first, second)
        ref_R = torch.stack([first, second, third], dim=2)

        ref_gate_vec = torch.einsum(
            "nij,nj->ni", ref_R, ref_obs[:, :3]) * 10.0
        geo4 = (torch.gather(
            geo_off, 1, k4[..., None].expand(-1, -1, 3))
            * w4[..., None]).sum(dim=1)
        ref_gate_vec = ref_gate_vec - geo4
        cur_pos = self.demo_gate_position[g] - gate_vec_world
        ref_pos = self.demo_gate_position[g] - ref_gate_vec

        # gate-center funnel
        gate_n = self.gate_normal[g]
        plane_dist = torch.abs(((cur_pos - self.demo_gate_position[g])
                                * gate_n).sum(-1))
        funnel_w = torch.zeros(n, device=dev)
        if self.funnel_dist > 0.0:
            start = self.funnel_dist
            full = min(self.funnel_full, start - 1e-3)
            w = self.funnel_strength * torch.clamp(
                (start - plane_dist) / max(start - full, 1e-3), 0.0, 1.0)
            funnel_w = w * self.funnel_mask[g] * (
                plane_dist <= start).float()
            from_center = ref_pos - self.demo_gate_position[g]
            in_plane = from_center - gate_n * (
                (from_center * gate_n).sum(-1, keepdim=True))
            ref_pos = ref_pos - funnel_w[:, None] * in_plane

        pos_err_w = cur_pos - ref_pos
        cur_vel_w = v
        ref_vel_w = (self.demo_vel[rows4] * w4[..., None]).sum(dim=1)
        ref_acc_w = (self.demo_acc[rows4] * w4[..., None]).sum(dim=1)
        vsc = self.vel_scale[g]
        vel_err_w = cur_vel_w - ref_vel_w
        att_err = _rotvec_batch(
            torch.einsum("nji,njk->nik", R, ref_R))
        pos_err_ref = torch.einsum("nji,nj->ni", ref_R, pos_err_w)
        vel_err_ref = torch.einsum("nji,nj->ni", ref_R, vel_err_w)

        tracking = torch.zeros(n, ACT_DIM, device=dev)
        tracking[:, :3] = torch.clamp(
            att_err * torch.tensor([0.70, 0.70, 0.60], device=dev),
            -0.25, 0.25)
        lgs = self.lat_gain_scale[g]
        raw_lat = (-lgs * self.lat_pos_gain * pos_err_ref[:, 1]
                   - lgs * self.lat_vel_gain * vel_err_ref[:, 1])
        tracking[:, 0] = tracking[:, 0] + torch.clamp(raw_lat, -0.20,
                                                      0.20)
        tracking[:, 0] = torch.where(
            self.lat_bias[g] != 0.0,
            torch.clamp(tracking[:, 0] + self.lat_bias[g], -0.25, 0.25),
            tracking[:, 0])
        vgs = self.vert_gain_scale[g]
        tracking[:, 3] = torch.clamp(
            vgs * self.vert_pos_gain * pos_err_w[:, 2]
            + vgs * self.vert_vel_gain * vel_err_w[:, 2], -0.15, 0.15)
        tracking[:, 3] = torch.where(
            self.vert_bias[g] != 0.0,
            torch.clamp(tracking[:, 3] + self.vert_bias[g], -0.15, 0.15),
            tracking[:, 3])
        reference = reference + self.feedback_scale * tracking

        tsc = self.thrust_scale[g]
        thrust01 = 0.5 * (reference[:, 3] + 1.0)
        thrust01 = 0.25 + tsc * (thrust01 - 0.25)
        reference = torch.cat([
            reference[:, :3],
            torch.clamp(2.0 * thrust01 - 1.0, -1.0, 1.0)[:, None]],
            dim=1)

        # trajectory blend (RateController semantics, batched)
        blend = self.traj_blend[g]
        active = blend > 0.0
        a_cmd = (self.traj_kp * (ref_pos - cur_pos)
                 + self.traj_kv * (vsc[:, None] * ref_vel_w - cur_vel_w)
                 + vsc[:, None] ** 2 * ref_acc_w)
        zi_new = torch.clamp(
            self.traj_zi + 0.6 * (ref_pos[:, 2] - cur_pos[:, 2]) * DT,
            -3.0, 3.0)
        self.traj_zi = torch.where(active, zi_new, self.traj_zi)
        a_cmd = torch.cat([a_cmd[:, :2], (a_cmd[:, 2]
                                          + self.traj_zi)[:, None]],
                          dim=1)
        ah = torch.linalg.norm(a_cmd[:, :2], dim=1, keepdim=True)
        scale_h = torch.clamp(14.0 / torch.clamp(ah, min=1e-6), max=1.0)
        a_cmd = torch.cat([a_cmd[:, :2] * scale_h,
                           torch.clamp(a_cmd[:, 2], -18.0, 12.0)[:, None]],
                          dim=1)
        t_des = a_cmd - torch.tensor([0.0, 0.0, G], device=dev)
        t_des = torch.cat([t_des[:, :2],
                           torch.clamp(t_des[:, 2], max=-2.0)[:, None]],
                          dim=1)
        t_norm = torch.linalg.norm(t_des, dim=1, keepdim=True)
        zb_des = -t_des / t_norm
        yaw = torch.atan2(ref_R[:, 1, 0], ref_R[:, 0, 0])
        xc = torch.stack([torch.cos(yaw), torch.sin(yaw),
                          torch.zeros_like(yaw)], dim=1)
        yb = torch.linalg.cross(zb_des, xc)
        yb_n = torch.linalg.norm(yb, dim=1, keepdim=True)
        yb = torch.where(yb_n < 1e-6,
                         torch.tensor([0.0, 1.0, 0.0],
                                      device=dev).expand(n, 3),
                         yb / torch.clamp(yb_n, min=1e-6))
        xb = torch.linalg.cross(yb, zb_des)
        R_des = torch.stack([xb, yb, zb_des], dim=2)
        w_cmd = self.traj_katt * _rotvec_batch(
            torch.einsum("nji,njk->nik", R, R_des))
        w_cmd = torch.clamp(w_cmd, -self.traj_rate_limit,
                            self.traj_rate_limit)
        thrust = torch.clamp(t_norm.squeeze(1) / self.traj_k_thrust,
                             0.02, self.traj_thrust_limit)
        traj_action = torch.cat([
            torch.clamp(self.rate_cmd_sign * self.traj_rate_cmd_gain
                        * w_cmd / self.wire_rate_limit, -1.0, 1.0),
            torch.clamp(2.0 * thrust - 1.0, -1.0, 1.0)[:, None]],
            dim=1)
        traj_action = torch.clamp(traj_action, -1.0, 1.0)
        b = (blend * active.float())[:, None]
        reference = (1.0 - b) * reference + b * traj_action

        base = reference          # teacher_blend == 1.0 (asserted)

        # gate-routed residual actors on the env-built observation
        residual = torch.zeros(n, ACT_DIM, device=dev)
        res_mean = torch.zeros(n, ACT_DIM, device=dev)
        if self.primary is not None or self.secondary is not None:
            obs = (obs_override if obs_override is not None
                   else self._build_obs(p, v, R, prev_action, g))
            if self.primary is not None:
                actor, mean, std = self.primary
                normalized = torch.clamp((obs - mean) / std, -8.0, 8.0)
                res_mean = actor.deterministic(normalized)
            if self.secondary is not None:
                actor, mean, std = self.secondary
                normalized = torch.clamp((obs - mean) / std, -8.0, 8.0)
                sec = actor.deterministic(normalized)
                use_sec = self.secondary_mask[g][:, None] > 0.5
                res_mean = torch.where(use_sec, sec, res_mean)
            residual = torch.clamp(res_mean, -self.residual_clip,
                                   self.residual_clip)
            residual = residual * self.residual_mask[g][:, None]

        selected = torch.clamp(base + self.residual_scale * residual,
                               -1.0, 1.0)
        if debug:
            return selected, {
                "reference_row": self.cursor.clone(),
                "reference_rows4": rows4.clone(),
                "reference_weights4": w4.clone(),
                "reference_observation": ref_obs.clone(),
                "reference_rotation": ref_R.clone(),
                "current_position": cur_pos.clone(),
                "reference_position": ref_pos.clone(),
                "current_velocity": cur_vel_w.clone(),
                "reference_velocity": ref_vel_w.clone(),
                "trajectory_rotation": R_des.clone(),
                "trajectory_action": traj_action.clone(),
                "reference_pre_residual": base.clone(),
                "tracking": tracking,
                "traj_blend": blend,
                "residual": residual,
                "res_mean": res_mean,
                "raw_lateral_feedback": raw_lat,
                "lat_gain_scale": lgs,
                "vel_scale": vsc,
                "thrust_scale": tsc,
                "action_lead": lead,
                "funnel_weight": funnel_w,
                "seg_start": self.seg_start[g],
                "seg_end": self.seg_end[g],
                "pos_err_ref": pos_err_ref,
                "vel_err_ref": vel_err_ref,
            }
        return selected

    def _build_obs(self, p, v, R, prev_action, g):
        """53-D observation exactly as the env/live stack lays it out."""
        n = p.shape[0]
        dev = self.dev
        Rt = R.transpose(1, 2)
        rel_rows = []
        tan_rows = []
        for lookahead in range(3):
            gk = torch.clamp(g + lookahead, max=N_RACE_GATES - 1)
            rel = torch.einsum(
                "nij,nj->ni", Rt,
                torch.tensor(self.gate_pos_map, device=dev)[gk] - p)
            rel_rows.append(rel / 10.0)
            tangent = self._gate_tangent[gk]
            tan_rows.append(torch.einsum("nij,nj->ni", Rt, tangent))
        v_body = torch.einsum("nij,nj->ni", Rt, v) / 10.0
        onehot = torch.zeros(n, N_RACE_GATES, device=dev)
        onehot[torch.arange(n, device=dev), g] = 1.0
        conf = torch.zeros(n, 2, device=dev)
        conf[:, 0] = 0.2      # nominal reported sigma/0.5
        conf[:, 1] = g.float() / (N_RACE_GATES - 1)
        return torch.cat(
            rel_rows + tan_rows
            + [v_body, R[:, :, 0], R[:, :, 1],
               torch.zeros(n, 3, device=dev),      # rates (approx 0 obs)
               prev_action, onehot, conf], dim=1)

    @property
    def _gate_tangent(self):
        if not hasattr(self, "_tan_cache"):
            pos = self.gate_pos_map
            tangents = np.zeros_like(pos)
            for gi in range(N_RACE_GATES):
                a = pos[max(0, gi - 1)]
                b = pos[min(N_RACE_GATES - 1, gi + 1)]
                d = b - a
                norm = np.linalg.norm(d)
                tangents[gi] = d / norm if norm > 1e-6 else (1, 0, 0)
            self._tan_cache = torch.tensor(
                tangents, dtype=torch.float32, device=self.dev)
        return self._tan_cache
