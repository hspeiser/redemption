"""Build the state observation vector from ground-truth odometry + the active gate pose.

Everything is expressed in the drone BODY frame so the policy is translation/heading invariant:
where the gate is relative to me, how I'm moving, how I'm rotating, which way is down, and which
way the gate faces. All ground truth — no perception noise.
"""
from __future__ import annotations
import numpy as np
from .geom import rot_world_to_body, gate_normal_world

_DOWN_WORLD = np.array([0.0, 0.0, 1.0])   # NED: +z is down


def dist_to_gate(pos, gate) -> float:
    return float(np.linalg.norm(gate["pos"] - pos))


def build_obs(mav, gate, cfg) -> np.ndarray:
    """mav: MavLink with live pos/quat/vel/rates. gate: dict with pos, quat."""
    q = mav.quat
    rel_world = gate["pos"] - mav.pos
    rel_body = rot_world_to_body(q, rel_world) / cfg.pos_scale
    vel_body = rot_world_to_body(q, mav.vel) / cfg.vel_scale
    rates = np.asarray(mav.rates, np.float64) / cfg.rate_scale
    grav_body = rot_world_to_body(q, _DOWN_WORLD)                 # unit-ish, encodes tilt
    normal_body = rot_world_to_body(q, gate_normal_world(gate["quat"]))
    return np.concatenate([rel_body, vel_body, rates, grav_body, normal_body]).astype(np.float32)
