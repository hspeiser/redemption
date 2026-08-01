"""Reference-line generation and tracking for the VQ2 line optimizer.

Produces a dynamically feasible 30 Hz reference (pos/vel/quat/action rows)
through the corrected map from a small parameter vector:

    offsets   (17,2)  crossing point in each gate's hole plane, bounded to
                      +/-(HOLE_HALF - clearance)
    seg_scale (18,)   fraction of the speed cap allowed on each course
                      segment (spawn->g0, g0->g1, ..., g15->g16)

Pipeline: Catmull-Rom spline through spawn + per-gate lead-in/cross/lead-out
points -> arc-length resample -> velocity profile (curvature, yaw-rate and
speed-cap limited, forward/backward accel passes) -> 30 Hz rows ->
differential-flatness attitude/rate/thrust feedforward in policy action
space.  FlatRefController tracks the rows inside FastVQ2Env as a backbone
(the same contract as refctl.RefController).

Limits are the live stack's: WIRE_RATE_LIMIT body rates, wire thrust cap
0.52, launch assist (min wire 0.30 while t<0.55s on gate 0) supplied by the
env itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from aigp.fastsim.env import HOLE_HALF, WIRE_RATE_LIMIT

N_GATES = 17
G = 9.81
DT = 1.0 / 30.0
# actual body-rate budget: the plant multiplies the wire command by the
# measured rate gain (K ~= 2.33/2.35/2.36), so a full-scale action yields
# 180/180/120 deg/s -- NOT the wire-unit WIRE_RATE_LIMIT numbers
RATE_ACTUAL = np.array([np.pi, np.pi, 2.0 * np.pi / 3.0])


def load_oriented_gates(map_path: str | Path):
    """Gate positions and frames oriented so local +y = course direction.

    Mirrors FastVQ2Env's orientation fixup exactly: the crossing test and
    the hole plane (local x,z) assume -y -> +y travel.
    """
    gates = json.loads(Path(map_path).read_text())["gates"]
    pos = np.array([g["pos"] for g in gates[:N_GATES]], float)
    quats = np.array([g["quat_wxyz"] for g in gates[:N_GATES]], float)
    R = Rotation.from_quat(
        np.stack([quats[:, 1], quats[:, 2], quats[:, 3], quats[:, 0]],
                 axis=1)
    ).as_matrix()
    for i in range(N_GATES):
        incoming = pos[i] - (pos[i - 1] if i else np.zeros(3))
        if np.dot(R[i, :, 1], incoming) < 0:
            R[i, :, 0] = -R[i, :, 0]
            R[i, :, 1] = -R[i, :, 1]
    return pos, R


@dataclass
class LineConfig:
    speed_cap: float = 8.0          # env kills above this; plan below it
    clearance: float = 0.25         # planned min distance to hole edge
    cap_margin: float = 0.90        # plan peak speed = cap * margin
    a_lat_max: float = 9.0          # curvature speed limit
    a_fwd: float = 5.0              # forward accel in profile pass
    a_brk: float = 6.0              # braking decel in profile pass
    yaw_margin: float = 0.70        # fraction of wire yaw-rate budget
    normal_lead_m: float = 1.3      # lead-in/out points along gate normal
    ds: float = 0.20                # arc-length sample spacing
    launch_speed: float = 0.5       # profile floor speed at t=0
    spawn: tuple = (0.0, 0.0, -0.05)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    """Centered moving average along axis 0 with edge replication."""
    if w <= 1:
        return x
    pad = w // 2
    xp = np.concatenate([np.repeat(x[:1], pad, 0), x,
                         np.repeat(x[-1:], pad, 0)], axis=0)
    kernel = np.ones(w) / w
    if x.ndim == 1:
        return np.convolve(xp, kernel, mode="valid")[:len(x)]
    out = np.stack([np.convolve(xp[:, k], kernel, mode="valid")[:len(x)]
                    for k in range(x.shape[1])], axis=1)
    return out


def _catmull_points(ctrl: np.ndarray, ds: float) -> np.ndarray:
    """Chord-length Catmull-Rom through ctrl points, resampled ~ds."""
    t = np.zeros(len(ctrl))
    seg = np.linalg.norm(np.diff(ctrl, axis=0), axis=1)
    t[1:] = np.cumsum(np.maximum(seg, 1e-6))
    tang = np.zeros_like(ctrl)
    tang[1:-1] = (ctrl[2:] - ctrl[:-2]) / (t[2:] - t[:-2])[:, None]
    tang[0] = (ctrl[1] - ctrl[0]) / (t[1] - t[0])
    tang[-1] = (ctrl[-1] - ctrl[-2]) / (t[-1] - t[-2])
    spl = CubicHermiteSpline(t, ctrl, tang)
    dense = spl(np.linspace(0.0, t[-1], max(int(t[-1] / 0.02), 200)))
    # resample by true arc length
    d = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    n_out = max(int(s[-1] / ds), 50)
    si = np.linspace(0.0, s[-1], n_out)
    out = np.stack([np.interp(si, s, dense[:, k]) for k in range(3)], axis=1)
    return out


def build_reference(
    gate_pos: np.ndarray,
    gate_R: np.ndarray,
    offsets: np.ndarray,
    seg_scale: np.ndarray,
    cfg: LineConfig,
) -> dict:
    """Parameter vector -> 30 Hz reference rows + diagnostics."""
    off_lim = HOLE_HALF - cfg.clearance
    off = np.clip(np.asarray(offsets, float).reshape(N_GATES, 2),
                  -off_lim, off_lim)
    scale = np.clip(np.asarray(seg_scale, float).reshape(N_GATES + 1),
                    0.35, 1.0)

    cross = gate_pos + off[:, 0:1] * gate_R[:, :, 0] \
        + off[:, 1:2] * gate_R[:, :, 2]
    normal = gate_R[:, :, 1]

    # control sequence: spawn, then lead-in / crossing / lead-out per gate
    ctrl = [np.asarray(cfg.spawn, float)]
    ctrl_gate = [-1]                      # segment id at each ctrl point
    for i in range(N_GATES):
        prev = ctrl[-1]
        nxt = cross[i + 1] if i + 1 < N_GATES else None
        lead = cfg.normal_lead_m
        lead_in = min(lead, 0.35 * np.linalg.norm(cross[i] - prev))
        lead_out = lead if nxt is None else min(
            lead, 0.35 * np.linalg.norm(nxt - cross[i]))
        ctrl.append(cross[i] - lead_in * normal[i])
        ctrl_gate.append(i)
        ctrl.append(cross[i])
        ctrl_gate.append(i)
        ctrl.append(cross[i] + lead_out * normal[i])
        ctrl_gate.append(i + 1)
    ctrl = np.asarray(ctrl)

    path = _catmull_points(ctrl, cfg.ds)
    n = len(path)
    d = np.gradient(path, axis=0)
    ds_arr = np.linalg.norm(d, axis=1)
    tan = d / (ds_arr[:, None] + 1e-9)
    s = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(path, axis=0), axis=1))])

    # gate stations along s (nearest sample to each crossing point)
    station = np.empty(N_GATES)
    lo = 0
    for i in range(N_GATES):
        dist = np.linalg.norm(path[lo:] - cross[i], axis=1)
        k = lo + int(np.argmin(dist))
        station[i] = s[k]
        lo = k
    # segment id per sample
    seg_id = np.searchsorted(station, s, side="right")

    # curvature and horizontal heading rate per unit length (smoothed:
    # the closely spaced gate lead points create local curvature spikes
    # that would otherwise inject single-row braking demands)
    dtan = np.gradient(tan, axis=0) / (ds_arr[:, None] + 1e-9)
    kappa = _smooth(np.linalg.norm(dtan, axis=1), 7)
    psi = np.unwrap(np.arctan2(tan[:, 1], tan[:, 0]))
    dpsi = _smooth(np.abs(np.gradient(psi)) / (ds_arr + 1e-9), 7)

    yaw_budget = cfg.yaw_margin * float(RATE_ACTUAL[2])
    v_cap = cfg.speed_cap * cfg.cap_margin * scale[seg_id]
    v_lim = np.minimum(v_cap, np.sqrt(cfg.a_lat_max /
                                      np.maximum(kappa, 1e-4)))
    v_lim = np.minimum(v_lim, yaw_budget / np.maximum(dpsi, 1e-4))
    v_lim = np.maximum(v_lim, 0.8)        # never fully stall mid-course

    # forward/backward passes on v^2
    v = v_lim.copy()
    v[0] = cfg.launch_speed
    for k in range(1, n):
        step = s[k] - s[k - 1]
        v[k] = min(v[k], np.sqrt(v[k - 1] ** 2 + 2 * cfg.a_fwd * step))
    for k in range(n - 2, -1, -1):
        step = s[k + 1] - s[k]
        v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2 * cfg.a_brk * step))

    # time integration -> 30 Hz rows
    seg_dt = np.diff(s) / np.maximum(0.5 * (v[1:] + v[:-1]), 1e-3)
    t_of_s = np.concatenate([[0.0], np.cumsum(seg_dt)])
    lap_t = float(t_of_s[-1])
    n_rows = int(lap_t / DT) + 1
    tr = np.arange(n_rows) * DT
    sr = np.interp(tr, t_of_s, s)
    pos_r = np.stack([np.interp(sr, s, path[:, k]) for k in range(3)], 1)
    vr = np.interp(sr, s, v)
    tan_r = np.stack([np.interp(sr, s, tan[:, k]) for k in range(3)], 1)
    tan_r /= np.linalg.norm(tan_r, axis=1, keepdims=True) + 1e-9
    vel_r = tan_r * vr[:, None]

    # gate crossing times / per-row target index
    t_gate = np.interp(station, s, t_of_s)
    target_r = np.searchsorted(t_gate, tr, side="right")

    # flatness feedforward: accel -> thrust vector -> attitude.
    # Smooth first, and never demand more than half-g downward accel:
    # near free-fall the thrust direction is undefined and the attitude
    # sequence flips (the feedback loops absorb the difference).
    vel_s = _smooth(vel_r, 5)
    acc = _smooth(np.gradient(vel_s, DT, axis=0), 5)
    acc = np.clip(acc, -25.0, 25.0)
    acc[:, 2] = np.minimum(acc[:, 2], 0.5 * G)    # z-down: +z = downward
    tvec = acc - np.array([0.0, 0.0, G])          # = R @ (-T z_b)
    T = np.linalg.norm(tvec, axis=1)
    T = np.maximum(T, 3.0)
    z_b = -tvec / T[:, None]

    # yaw: smoothed horizontal course heading (the gate lead-in points
    # already align the path with each gate normal, so tangent yaw keeps
    # the active gate inside the camera frustum without the step
    # discontinuity of a look-at-target scheme)
    hspeed = np.linalg.norm(tan_r[:, :2], axis=1)
    psi_r = np.arctan2(tan_r[:, 1], tan_r[:, 0])
    slow = hspeed < 0.3
    if slow.any() and not slow.all():
        first_ok = int(np.argmax(~slow))
        psi_r[:first_ok] = psi_r[first_ok]
    psi_r = _smooth(np.unwrap(psi_r), 15)
    look = np.stack([np.cos(psi_r), np.sin(psi_r),
                     np.zeros_like(psi_r)], axis=1)
    x_b = look - (look * z_b).sum(1, keepdims=True) * z_b
    bad = np.linalg.norm(x_b, axis=1) < 1e-3
    x_b[bad] = np.array([1.0, 0.0, 0.0])
    x_b /= np.linalg.norm(x_b, axis=1, keepdims=True)
    y_b = np.cross(z_b, x_b)
    R_raw = np.stack([x_b, y_b, z_b], axis=2)     # columns x,y,z

    # slew-limit the attitude sequence to the wire rate budget so the
    # feedforward is always actually flyable; feedback covers the rest
    rot_raw = Rotation.from_matrix(R_raw)
    step_max = 0.90 * RATE_ACTUAL * DT
    quats = [rot_raw[0]]
    for k in range(1, n_rows):
        stp = (quats[-1].inv() * rot_raw[k]).as_rotvec()
        stp = np.clip(stp, -step_max, step_max)
        quats.append(quats[-1] * Rotation.from_rotvec(stp))
    rot = Rotation.concatenate(quats)
    R_r = rot.as_matrix()

    # body rates from consecutive attitudes
    dq = (rot[:-1].inv() * rot[1:]).as_rotvec() / DT
    rates = np.vstack([dq, dq[-1:]])

    # thrust consistent with the (slew-limited) attitude actually flown
    T = np.maximum(-(tvec * R_r[:, :, 2]).sum(1), 3.0)

    return {
        "pos": pos_r.astype(np.float32),
        "vel": vel_r.astype(np.float32),
        "acc": acc.astype(np.float32),
        "psi": psi_r.astype(np.float32),
        "R": R_r.astype(np.float32),
        "quat_wxyz": np.roll(rot.as_quat(), 1, axis=1).astype(np.float32),
        "rates": rates.astype(np.float32),
        "thrust_acc": T.astype(np.float32),
        "gate": target_r.astype(np.int64),
        "t": tr.astype(np.float32),
        "cross": cross.astype(np.float32),
        "t_gate": t_gate.astype(np.float32),
        "planned_lap_s": lap_t,
        "offsets": off.astype(np.float32),
        "seg_scale": scale.astype(np.float32),
    }


def feedforward_actions(ref: dict, model) -> np.ndarray:
    """Rates+thrust rows -> policy-space [-1,1] actions (act-frame)."""
    g1 = float(model.thrust_gain)
    g2 = float(model.thrust_quad)
    T = ref["thrust_acc"].astype(float)
    wire = (-g1 + np.sqrt(g1 * g1 + 4.0 * g2 * T)) / (2.0 * g2)
    wire = np.clip(wire, 0.02, 0.52)
    act = np.empty((len(T), 4), np.float32)
    act[:, :3] = np.clip(ref["rates"] / RATE_ACTUAL, -1.0, 1.0)
    act[:, 3] = 2.0 * wire - 1.0
    return act


def feasibility(ref: dict) -> dict:
    """Fraction of rows where feedforward saturates the wire limits."""
    r = np.abs(ref["rates"]) / RATE_ACTUAL
    return {
        "rate_sat_frac": float((r > 0.90).any(axis=1).mean()),
        "rate_peak_frac": float(r.max()),
        "thrust_peak": float(ref["thrust_acc"].max()),
        "planned_lap_s": float(ref["planned_lap_s"]),
        "speed_peak": float(np.linalg.norm(ref["vel"], axis=1).max()),
    }


class FlatRefController:
    """RefController-compatible geometric (SE(3)-style) tracker.

    Open-loop rate feedforward diverges the moment the drone's timeline
    slips from the reference's, so every step recomputes the desired
    acceleration (ff + position/velocity PD), converts it to a desired
    attitude at the reference yaw, and commands body rates from the
    attitude error plus the ff rates.  Thrust is the desired specific
    force projected on the CURRENT body z-axis, inverted through the
    measured motor curve.
    """

    def __init__(self, ref: dict, ff_act: np.ndarray, n_envs: int,
                 device: str = "cpu", speed_cap: float = 1e9,
                 model=None,
                 kp: float = 2.0, kv: float = 2.8, katt: float = 5.0,
                 max_advance: int = 8, max_retreat: int = 2, lead: int = 4):
        dev = torch.device(device)
        self.speed_cap = float(speed_cap)
        self.g1 = float(model.thrust_gain) if model is not None else 14.05
        self.g2 = float(model.thrust_quad) if model is not None else 64.86
        self.kp, self.kv, self.katt = kp, kv, katt
        tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32,
                                    device=dev)
        self.P = tt(ref["pos"])
        self.V = tt(ref["vel"])
        self.ACC = tt(ref["acc"])
        self.PSI = tt(ref["psi"])
        self.W = tt(ref["rates"])
        n_pts = len(self.P)
        speed = torch.linalg.norm(self.V, dim=1)
        past = torch.nonzero(speed > 1.5).squeeze(-1)
        self.launch_rows = int(past[0]) if len(past) else 40
        self.idx = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.steps = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.n_envs = n_envs
        self.n_pts = n_pts
        self.dev = dev
        self.max_advance = max_advance
        self.max_retreat = max_retreat
        self.lead = lead
        self.rate_scale = tt(RATE_ACTUAL)
        self.g_vec = tt([0.0, 0.0, G])

    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.steps[env_ids] = 0

    @torch.no_grad()
    def action(self, p: torch.Tensor, v: torch.Tensor,
               R: torch.Tensor) -> torch.Tensor:
        n = p.shape[0]
        offs = torch.arange(-self.max_retreat, self.max_advance + 1,
                            device=self.dev)
        cand = torch.clamp(self.idx[:, None] + offs[None, :], 0,
                           self.n_pts - 1)
        dif = self.P[cand] - p[:, None, :]
        dist = torch.linalg.norm(dif, dim=-1)
        best = dist.argmin(dim=1)
        nearest = cand[torch.arange(n, device=self.dev), best]
        self.steps = self.steps + 1
        self.idx = nearest
        # scripted launch: chase a time-advancing row until the tracker
        # is genuinely airborne and moving
        in_launch = nearest < self.launch_rows
        k = torch.clamp(self.idx + self.lead, max=self.n_pts - 1)
        floor_row = torch.clamp(self.steps + 5, max=self.launch_rows)
        k = torch.where(in_launch, floor_row.long(), k)

        # desired specific force: ff accel + PD, minus gravity (z-down)
        e_p = torch.clamp(self.P[k] - p, -3.0, 3.0)
        e_v = torch.clamp(self.V[k] - v, -4.0, 4.0)
        # drag compensation (nominal measured aero: ~0.30/s linear +
        # ~0.03/m quadratic); without it the speed equilibrium sits
        # ~1.2 m/s under the reference profile
        drag = 0.30 * v + 0.030 * torch.linalg.norm(
            v, dim=-1, keepdim=True) * v
        a_des = self.ACC[k] + drag + self.kp * e_p + self.kv * e_v
        a_des[:, 2] = torch.clamp(a_des[:, 2], max=0.5 * G)
        tvec = a_des - self.g_vec                    # = T * (-z_b_des)
        T_norm = torch.linalg.norm(tvec, dim=-1).clamp(min=3.0)
        z_des = -tvec / T_norm[:, None]

        psi = self.PSI[k]
        look = torch.stack([torch.cos(psi), torch.sin(psi),
                            torch.zeros_like(psi)], dim=1)
        x_des = look - (look * z_des).sum(-1, keepdim=True) * z_des
        x_des = torch.nn.functional.normalize(x_des, dim=1, eps=1e-6)
        y_des = torch.linalg.cross(z_des, x_des)
        R_des = torch.stack([x_des, y_des, z_des], dim=2)

        # attitude error rotvec (body frame): log(R^T R_des)
        E = torch.einsum("nij,njk->nik", R.transpose(1, 2), R_des)
        trace = E[:, 0, 0] + E[:, 1, 1] + E[:, 2, 2]
        cos_a = torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0)
        ang = torch.arccos(cos_a)
        vee = torch.stack([E[:, 2, 1] - E[:, 1, 2],
                           E[:, 0, 2] - E[:, 2, 0],
                           E[:, 1, 0] - E[:, 0, 1]], dim=1)
        sin_a = torch.sin(ang).clamp(min=1e-4)
        rotvec = vee * (ang / (2.0 * sin_a))[:, None]

        rate_cmd = self.W[k] + self.katt * rotvec
        act_rates = torch.clamp(rate_cmd / self.rate_scale, -1.0, 1.0)

        # thrust: desired force projected on the actual body z-axis
        T_cmd = (-(tvec) * R[:, :, 2]).sum(-1).clamp(min=2.0, max=26.0)
        wire = (-self.g1 + torch.sqrt(
            self.g1 * self.g1 + 4.0 * self.g2 * T_cmd)) / (2.0 * self.g2)
        # overspeed governor: the env kills above the cap
        speed = torch.linalg.norm(v, dim=-1)
        over = torch.clamp(
            (speed - 0.88 * self.speed_cap) / (0.12 * self.speed_cap),
            0.0, 1.0,
        )
        wire = wire - 0.20 * over
        wire = torch.where(in_launch, wire.clamp(min=0.36), wire)
        a3 = 2.0 * torch.clamp(wire, 0.02, 0.52) - 1.0
        return torch.cat([act_rates, a3[:, None]], dim=1)


class BatchedFlatRefController:
    """FlatRefController over a POPULATION of references at once.

    Env e tracks candidate e // envs_per_cand.  References are padded to
    the longest row count (repeating the final row), so the whole
    population flies in a single FastVQ2Env batch and the GPU stays fed.
    Same control law as FlatRefController.
    """

    def __init__(self, refs: list, n_per: int, device: str = "cpu",
                 speed_cap: float = 1e9, model=None,
                 kp: float = 2.0, kv: float = 2.8, katt: float = 5.0,
                 max_advance: int = 8, max_retreat: int = 2, lead: int = 4):
        dev = torch.device(device)
        C = len(refs)
        n_envs = C * n_per
        self.speed_cap = float(speed_cap)
        self.g1 = float(model.thrust_gain) if model is not None else 14.05
        self.g2 = float(model.thrust_quad) if model is not None else 64.86
        self.kp, self.kv, self.katt = kp, kv, katt
        N = max(len(r["pos"]) for r in refs)

        def pad(key, width):
            out = np.zeros((C, N, width) if width > 1 else (C, N),
                           np.float32)
            for c, r in enumerate(refs):
                a = r[key]
                out[c, :len(a)] = a
                out[c, len(a):] = a[-1]
            return torch.tensor(out, device=dev)

        self.P = pad("pos", 3)
        self.V = pad("vel", 3)
        self.ACC = pad("acc", 3)
        self.PSI = pad("psi", 1)
        self.W = pad("rates", 3)
        self.n_pts = torch.tensor(
            [len(r["pos"]) for r in refs], device=dev)
        launch = []
        for r in refs:
            speed = np.linalg.norm(r["vel"], axis=1)
            past = np.nonzero(speed > 1.5)[0]
            launch.append(int(past[0]) if len(past) else 40)
        self.launch_rows = torch.tensor(launch, device=dev)
        self.cand = torch.arange(n_envs, device=dev) // n_per
        self.idx = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.steps = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.n_envs = n_envs
        self.dev = dev
        self.max_advance = max_advance
        self.max_retreat = max_retreat
        self.lead = lead
        self.rate_scale = torch.tensor(RATE_ACTUAL, dtype=torch.float32,
                                       device=dev)
        self.g_vec = torch.tensor([0.0, 0.0, G], device=dev)

    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.steps[env_ids] = 0

    def reset_nearest(self, env_ids, p):
        # every reference starts at spawn; a fresh env starts at row 0
        self.idx[env_ids] = 0
        self.steps[env_ids] = 0

    @torch.no_grad()
    def action(self, p: torch.Tensor, v: torch.Tensor,
               R: torch.Tensor) -> torch.Tensor:
        n = p.shape[0]
        c = self.cand
        last = self.n_pts[c] - 1
        offs = torch.arange(-self.max_retreat, self.max_advance + 1,
                            device=self.dev)
        cand_rows = torch.clamp(
            self.idx[:, None] + offs[None, :], torch.zeros_like(
                last)[:, None], last[:, None])
        dif = self.P[c[:, None], cand_rows] - p[:, None, :]
        best = torch.linalg.norm(dif, dim=-1).argmin(dim=1)
        nearest = cand_rows[torch.arange(n, device=self.dev), best]
        self.steps = self.steps + 1
        self.idx = nearest
        in_launch = nearest < self.launch_rows[c]
        k = torch.minimum(self.idx + self.lead, last)
        floor_row = torch.minimum(self.steps + 5, self.launch_rows[c])
        k = torch.where(in_launch, floor_row, k)

        e_p = torch.clamp(self.P[c, k] - p, -3.0, 3.0)
        e_v = torch.clamp(self.V[c, k] - v, -4.0, 4.0)
        drag = 0.30 * v + 0.030 * torch.linalg.norm(
            v, dim=-1, keepdim=True) * v
        a_des = self.ACC[c, k] + drag + self.kp * e_p + self.kv * e_v
        a_des[:, 2] = torch.clamp(a_des[:, 2], max=0.5 * G)
        tvec = a_des - self.g_vec
        T_norm = torch.linalg.norm(tvec, dim=-1).clamp(min=3.0)
        z_des = -tvec / T_norm[:, None]

        psi = self.PSI[c, k]
        look = torch.stack([torch.cos(psi), torch.sin(psi),
                            torch.zeros_like(psi)], dim=1)
        x_des = look - (look * z_des).sum(-1, keepdim=True) * z_des
        x_des = torch.nn.functional.normalize(x_des, dim=1, eps=1e-6)
        y_des = torch.linalg.cross(z_des, x_des)
        R_des = torch.stack([x_des, y_des, z_des], dim=2)

        E = torch.einsum("nij,njk->nik", R.transpose(1, 2), R_des)
        trace = E[:, 0, 0] + E[:, 1, 1] + E[:, 2, 2]
        ang = torch.arccos(torch.clamp(0.5 * (trace - 1.0), -1.0, 1.0))
        vee = torch.stack([E[:, 2, 1] - E[:, 1, 2],
                           E[:, 0, 2] - E[:, 2, 0],
                           E[:, 1, 0] - E[:, 0, 1]], dim=1)
        sin_a = torch.sin(ang).clamp(min=1e-4)
        rotvec = vee * (ang / (2.0 * sin_a))[:, None]

        rate_cmd = self.W[c, k] + self.katt * rotvec
        act_rates = torch.clamp(rate_cmd / self.rate_scale, -1.0, 1.0)

        T_cmd = (-(tvec) * R[:, :, 2]).sum(-1).clamp(min=2.0, max=26.0)
        wire = (-self.g1 + torch.sqrt(
            self.g1 * self.g1 + 4.0 * self.g2 * T_cmd)) / (2.0 * self.g2)
        speed = torch.linalg.norm(v, dim=-1)
        over = torch.clamp(
            (speed - 0.88 * self.speed_cap) / (0.12 * self.speed_cap),
            0.0, 1.0,
        )
        wire = wire - 0.20 * over
        wire = torch.where(in_launch, wire.clamp(min=0.36), wire)
        a3 = 2.0 * torch.clamp(wire, 0.02, 0.52) - 1.0
        return torch.cat([act_rates, a3[:, None]], dim=1)


def demo_states_from_pop(refs: list) -> dict:
    """Union corridor + spawn source over a candidate population."""
    return {
        "pos": np.concatenate([r["pos"][::3] for r in refs], axis=0),
        "vel": np.concatenate([r["vel"][::3] for r in refs], axis=0),
        "quat": np.concatenate([r["quat_wxyz"][::3] for r in refs],
                               axis=0),
        "gate": np.concatenate(
            [r["gate"][::3].astype(np.float32) for r in refs], axis=0),
    }


def demo_states_from_ref(ref: dict) -> dict:
    """FastVQ2Env demo_states dict (corridor + start curriculum source)."""
    return {
        "pos": ref["pos"],
        "vel": ref["vel"],
        "quat": ref["quat_wxyz"],
        "gate": ref["gate"].astype(np.float32),
    }
