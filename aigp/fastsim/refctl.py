"""Reference-line backbone controller (the hybrid's dependable half).

Follows the certified winning lap: feedforward = the lap's logged
policy-space actions at the nearest reference row, feedback = PD on
path-frame position/velocity error projected onto body axes.  This is
the same recipe as the SAC campaign's teacher (which passes gate 1 at
96%), reimplemented vectorized for fastsim training and reusable as a
single-instance live backbone.

The learned residual rides on top:
    action = clamp(backbone + residual_scale * pi(obs), -1, 1)
"""

from __future__ import annotations

import numpy as np
import torch


class RefController:
    def __init__(
        self,
        path_pos: np.ndarray,          # (N,3)
        path_vel: np.ndarray,          # (N,3)
        ff_actions: np.ndarray,        # (N,4) policy-space [-1,1]
        n_envs: int,
        device: str = "cpu",
        # v77 teacher's proven feedback gains (action units per meter /
        # per m/s)
        lat_pos_gain: float = 0.10,
        lat_vel_gain: float = 0.30,
        vert_pos_gain: float = 0.20,
        vert_vel_gain: float = 0.30,
        along_vel_gain: float = 0.0,
        max_advance: int = 8,
        max_retreat: int = 2,
        lead: int = 4,
    ):
        dev = torch.device(device)
        self.P = torch.tensor(path_pos, dtype=torch.float32, device=dev)
        self.V = torch.tensor(path_vel, dtype=torch.float32, device=dev)
        self.A = torch.tensor(ff_actions, dtype=torch.float32,
                              device=dev).clamp(-1, 1)
        n_pts = len(self.P)
        # path tangents (unit) and per-row lateral/vertical frames
        d = torch.zeros_like(self.P)
        d[1:-1] = self.P[2:] - self.P[:-2]
        d[0] = self.P[1] - self.P[0]
        d[-1] = self.P[-1] - self.P[-2]
        self.T = torch.nn.functional.normalize(d, dim=1)
        up = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(n_pts, 3)
        lat = torch.linalg.cross(self.T, up)
        self.L = torch.nn.functional.normalize(lat, dim=1)
        self.U = torch.nn.functional.normalize(
            torch.linalg.cross(self.L, self.T), dim=1)
        self.idx = torch.zeros(n_envs, dtype=torch.long, device=dev)
        # launch bootstrap: the reference itself starts at rest, so a
        # purely position-indexed replay deadlocks on the pad; advance
        # the index with time through the launch segment
        self.steps = torch.zeros(n_envs, dtype=torch.long, device=dev)
        self.launch_rows = 55
        self.n_envs = n_envs
        self.n_pts = n_pts
        self.dev = dev
        self.g = dict(lp=lat_pos_gain, lv=lat_vel_gain,
                      vp=vert_pos_gain, vv=vert_vel_gain,
                      av=along_vel_gain)
        self.max_advance = max_advance
        self.max_retreat = max_retreat
        self.lead = lead

    def reset(self, env_ids):
        self.idx[env_ids] = 0
        self.steps[env_ids] = 0

    @torch.no_grad()
    def action(self, p: torch.Tensor, v: torch.Tensor,
               R: torch.Tensor) -> torch.Tensor:
        """p (n,3) belief position, v (n,3) belief velocity,
        R (n,3,3) body->world.  Returns (n,4) policy-space action."""
        n = p.shape[0]
        # windowed nearest-point index advance
        offs = torch.arange(-self.max_retreat, self.max_advance + 1,
                            device=self.dev)
        cand = torch.clamp(self.idx[:, None] + offs[None, :], 0,
                           self.n_pts - 1)                     # (n,W)
        dif = self.P[cand] - p[:, None, :]                     # (n,W,3)
        dist = torch.linalg.norm(dif, dim=-1)
        best = dist.argmin(dim=1)
        nearest = cand[torch.arange(n, device=self.dev), best]
        self.steps = self.steps + 1
        self.idx = nearest
        # scripted launch phase: the demo's early rows are countdown
        # zeros and its launch thrust sits at the surrogate's hover
        # point, so replay cannot lift off.  Chase a time-advancing
        # reference row at strong climb thrust until the tracker is
        # genuinely past the launch segment.
        in_launch = nearest < self.launch_rows
        k = torch.clamp(self.idx + self.lead, max=self.n_pts - 1)
        floor_row = torch.clamp(self.steps + 5, max=self.launch_rows)
        k = torch.where(in_launch, floor_row.long(), k)

        err = self.P[k] - p                                    # (n,3)
        verr = self.V[k] - v
        e_lat = (err * self.L[k]).sum(-1)
        e_vert = (err * self.U[k]).sum(-1)
        ev_lat = (verr * self.L[k]).sum(-1)
        ev_vert = (verr * self.U[k]).sum(-1)
        ev_along = (verr * self.T[k]).sum(-1)

        # lateral velocity loop (same design as vertical): desired
        # lateral velocity from position error, roll correction from
        # velocity error, projected onto the body roll axis
        vlat_meas = (v * self.L[k]).sum(-1)
        vlat_ref = (self.V[k] * self.L[k]).sum(-1)
        vlat_des = torch.clamp(vlat_ref + 1.6 * e_lat, -2.2, 2.2)
        corr_world = (
            (0.35 * (vlat_des - vlat_meas))[:, None] * self.L[k]
        )
        body_y = R[:, :, 1]
        d_roll = torch.clamp((corr_world * body_y).sum(-1), -0.5, 0.5)
        # vertical: self-contained altitude-velocity loop around the
        # model hover point (v2 hover wire 0.295 -> action -0.41).
        # The demo's thrust rows were assist/hover-carried and replay
        # near the surrogate's hover margin -- unusable as feedforward.
        climb_meas = -v[:, 2]                       # z-down: -vz = climb
        climb_ref = -self.V[k][:, 2]
        vdes = torch.clamp(climb_ref + 1.6 * e_vert, -2.5, 2.5)
        a_thrust = -0.41 + 0.30 * (vdes - climb_meas)
        a = self.A[k].clone()
        a[:, 0] = a[:, 0] + d_roll
        a[:, 3] = a_thrust
        return torch.clamp(a, -1.0, 1.0)


def load_winner_backbone(demo_npz, episode_npz, n_envs, device="cpu",
                         **gains):
    demo = np.load(demo_npz)
    ep = np.load(episode_npz)
    pos = demo["pos"]
    vel = demo["vel"]
    # the episode's "action" key is the RESIDUAL head (~0.001 on the
    # winning lap -- the teacher flew it); the flown command is
    # wire_action: rates = a[:3]*WIRE_RATE_LIMIT, thrust01 = (a3+1)/2
    from aigp.fastsim.env import WIRE_RATE_LIMIT
    wire = ep["wire_action"]
    act = np.empty((len(wire), 4), np.float32)
    act[:, :3] = wire[:, :3] / np.asarray(WIRE_RATE_LIMIT, np.float32)
    act[:, 3] = 2.0 * wire[:, 3] - 1.0
    act = np.clip(act, -1.0, 1.0)
    n = min(len(pos), len(act))
    return RefController(pos[:n], vel[:n], act[:n], n_envs,
                         device=device, **gains)
